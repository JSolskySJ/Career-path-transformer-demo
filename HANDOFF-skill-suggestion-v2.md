# Implementation Spec — Skill Suggestions v2: Semantic Gate, Denominator Floor, Skill-Slot Fine-Tune, Sliced Eval

**Goal.** Fix the two known weaknesses of the counterfactual skill-suggestion scorer:
(1) high-lift semantic nonsense ("restful api implementation" suggested for a sales
manager), worst on thin/low-context profiles; (2) zero-shot miscalibration that makes
the shipped lift (B4) lose to plain popularity (B1) on LOSO recovery. Deliverables 1
and 3 are inference/eval-only and ship first; Deliverable 2 is the training follow-up.

**Read these before writing code:**
- `career-path-transformer-demo/demo/skill_suggestion.py` — demo-native two-pass
  scorer (`fast_scores`, `build_pass_rows`, `suggest`, `live_eval`). This is live in
  the demo (`/api/suggest_skills`, `/api/skill_eval`, `static/skills.html`).
- `datawarehouse-ai/models/career_path_skill_suggestion.py` — canonical at-scale
  implementation + runnable `__main__` self-checks (6 toy tests + real-checkpoint
  mode). **Currently UNCOMMITTED** on branch `DWR-7169_Career_Path_Transformer`.
- `datawarehouse-ai/evals/skill_suggestion_eval.py` — offline eval (B1–B4 baselines,
  MLflow). Also uncommitted.
- Reference maths: `~/Downloads/skill_delta_maths_by_hand.pptx`.

**Empirical baseline you are improving on** (50 random held-out profiles, R5 dataset
`d875e69d-…`, model R4 `dazzling-roo-991` / run `3fc79c63…`, which has
`sample_resumes.json` staged): 80% of verified suggestions positive delta, 38/40
top-1 positive, BUT thin profiles (0 skills) get 16–22× lifts on semantically random
skills, saturated profiles (target already #1–#5) get negative top-1s, and B1
popularity beats B4 on LOSO at interactive sample sizes.

---

## 1. Deliverable A — semantic gate + denominator floor (inference-only, no training)

### 1.1 Semantic gate
DenseRec checkpoints already carry an L2-normalised MiniLM content matrix:
`module.content` (vocab_size × 384) with a `module.has_content` mask (both repos'
module classes). Dot product of normalised rows = cosine.

- In candidate selection (both `fast_scores` in the demo and `suggest_skills_fast`
  in datawarehouse-ai): drop candidate skills with
  `cosine(content[skill_id], content[target_id]) < gate` **before** ranking by lift.
- `gate` is a parameter, default **0.25**; sweep {0.15, 0.25, 0.35} in the eval.
- Tokens without content (`has_content` false) and non-DenseRec checkpoints
  (bert4rec/modernbert have no content matrix): skip the gate for those candidates
  rather than erroring — but log/return how many candidates were gated vs passed
  through ungated, so the demo can display it.
- Surface the knob in `/api/suggest_skills` (`semantic_gate` float, 0 = off) and as
  a small input on `static/skills.html`.

### 1.2 Denominator floor / smoothing
Current: `lift = pA**alpha / clip(pB, 1e-9)` — a 1e-9 clip is not a floor, it's a
div-by-zero guard. Replace with add-k smoothing:
`lift = (pA + k)**alpha / (pB + k)`, with `k` default **1 / (10 * n_candidate_skills)**
(uniform-prior scale). Expose `k`; sweep {0, 1/(10n), 1/n} in the eval. Keep alpha.

### 1.3 Acceptance
- Module self-checks extended: (a) a candidate with low content-cosine to T is
  excluded when the gate is on and included when off; (b) with k>0, a skill with
  pB≈0 no longer produces an unbounded lift.
- Rerun the 50-profile spot analysis (see §4 recipe) — the "cross-domain nonsense"
  examples (e.g. restful api → sales manager) must disappear from top-5s.

## 2. Deliverable B — skill-slot fine-tune (datawarehouse-ai, training change)

This was explicitly deferred from v1; the zero-shot eval failing fidelity/LOSO is the
trigger, and that trigger has now fired.

### 2.1 Objective
Fine-tune the pretrained DenseRec checkpoint on rows shaped EXACTLY like the
inference rows from `build_pass_rows`:

```
[pad..., remaining_preamble_skills..., [MASK], experiences..., title_slot]
```

- Label at the `[MASK]`: one held-out preamble skill (sampled per worker per epoch).
- `title_slot`: the worker's held-out next title (the training target the sequence
  already has), **with title-dropout p=0.5** — half the batches replace it with
  `[MASK]`. This trains the Pass-B denominator distribution, not just Pass A.
- Loss: cross-entropy at the skill mask only, restricted to skill-token columns
  (softmax over skill ids, mirroring inference).

### 2.2 Where it plugs in
`models/career_path_transformer_denserec.py` already has fine-tune machinery
(`finetune_epochs`, `finetune_target`, `pretrained_run_id` for warm-starting from an
existing MLflow checkpoint — the transfer path the X-series used). Add a new
fine-tune mode (suggested flag: `finetune_target='skill_slot'`) with its own Dataset
class; do NOT touch the Cloze pretrain or the existing next-title fine-tune paths.
Note the existing code warns that fine-tuning drops W_DESC tokens — same limitation
is acceptable here, keep the warning accurate.

### 2.3 Must-handle
- **Oversample thin profiles.** Workers with ≤1 experience token or ≤1 remaining
  preamble skill after holdout must be ≥20% of fine-tune batches (weighted sampler).
  If thin profiles are filtered out upstream (min-length filters in
  `build_sequences`), the thin-context regime stays untrained and the main symptom
  survives.
- **Don't wreck title ranking.** Mix in standard Cloze batches (suggested 1:1) or
  freeze the head, and gate the run on test R@10 not regressing by more than 1 point
  vs the warm-start checkpoint. Log both objectives' losses separately to MLflow.
- Workers with zero preamble skills contribute no skill-slot rows (nothing to hold
  out) — they are inference-only consumers of the calibration, that's fine.
- `_PositionedSeq` / anchored runs: preamble slots (including the mask) anchor to
  career index 0; the title slot to max+1 — copy `_contexts_to_rows`' handling.
- Training kwargs come through `generate_models(**kwargs)` as airflow strings —
  follow the existing `_bool`/int coercion pattern, and add new params to
  `_model_params()` so they print and log (repo convention).

### 2.4 Acceptance
- Retrain gate: `evals/skill_suggestion_eval.py` on ≥2,000 held-out pairs with real
  training co-occurrence counts (`--cooc-json`): B4 must now beat B1 and B2 on LOSO
  MRR, fidelity overlap@20 ≥ 0.6, Spearman ≥ 0.7 (the v1 ship gates).
- Thin-slice specific: see Deliverable C metrics — thin-profile top-10 mean content
  cosine to target must rise materially vs the zero-shot checkpoint.

## 3. Deliverable C — context-richness slicing in both evals

- Define slices: **thin** = worker has 0 preamble skills OR < 5 in-vocab experience
  tokens; **rich** = everything else. One shared helper, same definition in both
  repos.
- `demo/skill_suggestion.live_eval`: report every existing metric (LOSO per
  baseline, genericity, fidelity, self-consistency) per slice + combined; add one
  new metric per slice: mean MiniLM content-cosine of the top-10 to the target
  ("semantic relatedness" — the number that catches nonsense directly).
- `static/skills.html` eval panel: render the per-slice table (thin / rich columns).
- `evals/skill_suggestion_eval.py`: same split + metric; keep MLflow logging.

## 4. Verification recipe (use throughout)

```bash
# env: conda dwh-ai-py311, KMP_DUPLICATE_LIB_OK=TRUE
# module self-checks (toy, seconds):
cd datawarehouse-ai && python -m models.career_path_skill_suggestion
# real checkpoint (R4, staged in the demo fetch cache):
python -m models.career_path_skill_suggestion \
  --model ../career-path-transformer-demo/incoming/model_3fc79c63.pth \
  --vocab ../career-path-transformer-demo/incoming/bert4rec_vocab_3fc79c63.json \
  --samples ../career-path-transformer-demo/artifacts/runs/3fc79c63173f42578415763358eb1834/sample_resumes.json
# demo live: CPT_AUTO_SYNC=0 CPT_DEMO_PORT=5055 ./run.sh, then /skills page +
# its Live evaluation panel. NEVER test on ports 5060/5061 (browser unsafe-port
# blocklist — connections refused, curl works, very confusing).
```

50-profile spot analysis: sample 50 of R4's `sample_resumes.json`, call
`demo.skill_suggestion.suggest` per profile with the held-out target, and check
(a) % positive verified deltas, (b) eyeball top-5 semantic fit for ~10 profiles,
(c) the thin-profile subset specifically.

## 5. Gotchas (all hit during v1 — do not rediscover them)

- The demo's staged `vocab.json` is a bare idx2str list; training counts live in
  the persistent fetch cache: `incoming/bert4rec_vocab_<run8>.json` → `counts`
  (`demo/skill_suggestion.load_counts`). Never assume `vocab.counts` is populated.
- The pass-row truncation rule (preamble survives, most-recent experiences fill)
  deliberately differs from the ranking path's tail-window — reuse
  `build_pass_rows`, don't reinvent.
- The two datawarehouse-ai files are uncommitted and the branch index holds OTHER
  staged work — never `git add -A` / commit the whole index there; commit only your
  own paths (`git commit -- <paths>`), or leave uncommitted and say so.
- MLflow tracking server is flaky (interrupted downloads, refused blobs — R5/R6
  checkpoints `chill-hawk-287`/`peaceful-dove-288` are undownloadable). Model
  loading always goes through the demo fetch script / its incoming cache.
- Demo eval co-occurrence is a proxy built from the sampled pairs; only the
  datawarehouse-ai eval with `--cooc-json` gives trustworthy B2/genericity numbers.
- Dataset quirk: education majors occasionally carry punctuation artifacts
  (e.g. `science)`) — harmless here, don't "fix" tokenisation in this task.

## 6. Wording + out of scope

- All user-facing copy stays correlational: "skills associated with workers who
  move into this role". For thin profiles add: "profile too thin to personalise —
  showing skills typical of workers entering this role".
- Out of scope: multi-skill greedy sets · OOV/external-taxonomy skill candidates ·
  gradient saliency · any UI redesign beyond the two knobs and the sliced eval
  table · causal framing of any output.
