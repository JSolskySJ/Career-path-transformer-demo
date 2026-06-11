"""Train a demo BERT4Rec checkpoint the app can load.

Why this exists: production BERT4Rec runs log only the torch model to MLflow —
the vocab is built in-memory at train time and never persisted — so no
production checkpoint is usable for local inference. This script trains the
same architecture on the real held-out career sequences found in the local
eval CSVs (context + correct_target = the candidate's full sequence) and saves
everything inference needs to artifacts/bert4rec/.

This is a demo-quality model (tens of thousands of sequences vs the ~5.5M the
production run sees) — it demonstrates the mechanism, not production recall.

Usage:
    python scripts/train_bert4rec.py [--csv PATH] [--epochs 40] [--device mps]
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # macOS torch/gensim OpenMP clash

import argparse
import copy
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from demo import config
from demo.bert4rec_model import BERT4Rec, Vocab
from scripts.data_common import load_sequences, newest_eval_csv


class SequenceDataset(Dataset):
    def __init__(self, sequences, vocab: Vocab, max_len: int):
        self.max_len = max_len
        self.pad_id  = vocab.pad_id
        self.sequences = [e for e in (vocab.encode(s) for s in sequences) if len(e) >= 2]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq    = self.sequences[idx][-self.max_len:]
        padded = [self.pad_id] * (self.max_len - len(seq)) + seq
        return torch.tensor(padded, dtype=torch.long)


def cloze_mask(batch, mask_id, pad_id, vocab_size, mask_prob):
    """BERT-style 80/10/10 masking, identical to the training repo."""
    is_pad   = batch == pad_id
    mask_pos = (torch.rand_like(batch, dtype=torch.float) < mask_prob) & ~is_pad
    targets  = torch.full_like(batch, -100)
    targets[mask_pos] = batch[mask_pos]
    masked   = batch.clone()
    r = torch.rand_like(batch, dtype=torch.float)
    masked[mask_pos & (r < 0.8)] = mask_id
    rand_replace = mask_pos & (r >= 0.8) & (r < 0.9)
    masked[rand_replace] = torch.randint(2, vocab_size, batch.shape, device=batch.device)[rand_replace]
    return masked, targets


def pick_device(name: str) -> torch.device:
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default=None, help='eval CSV (default: newest in model_eval_csv)')
    parser.add_argument('--epochs',     type=int,   default=40)
    parser.add_argument('--patience',   type=int,   default=5)
    parser.add_argument('--d-model',    type=int,   default=128)
    parser.add_argument('--n-layers',   type=int,   default=2)
    parser.add_argument('--n-heads',    type=int,   default=2)
    parser.add_argument('--max-len',    type=int,   default=64)
    parser.add_argument('--dropout',    type=float, default=0.2)
    parser.add_argument('--mask-prob',  type=float, default=0.3)
    parser.add_argument('--batch-size', type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--min-count',  type=int,   default=3)
    parser.add_argument('--seed',       type=int,   default=123)
    parser.add_argument('--device',     default='auto')
    args = parser.parse_args()

    if args.csv is None:
        args.csv = newest_eval_csv()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f'Device: {device}')

    seqs  = load_sequences(args.csv)
    vocab = Vocab.build(seqs, min_count=args.min_count)
    print(f'Vocab: {vocab.size:,} tokens')

    dataset = SequenceDataset(seqs, vocab, args.max_len)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f'Trainable sequences: {len(dataset):,}')

    model = BERT4Rec(
        vocab_size=vocab.size, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, max_len=args.max_len, dropout=args.dropout,
        pad_id=vocab.pad_id,
    ).to(device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_loss  = float('inf')
    best_state = None
    no_improve = 0
    stopped_at = args.epochs

    for epoch in range(args.epochs):
        model.train()
        total, n_batches = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            masked, targets = cloze_mask(batch, vocab.mask_id, vocab.pad_id,
                                         vocab.size, args.mask_prob)
            if not (targets != -100).any():
                continue
            loss = nn.functional.cross_entropy(
                model(masked).view(-1, vocab.size), targets.view(-1), ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_batches += 1
        avg = total / max(n_batches, 1)
        print(f'Epoch {epoch + 1}/{args.epochs}  loss={avg:.4f}', flush=True)
        if avg < best_loss:
            best_loss, no_improve = avg, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f'Early stopping at epoch {epoch + 1}')
                stopped_at = epoch + 1
                break

    if best_state:
        model.load_state_dict(best_state)

    out_dir = config.BERT4REC_DIR
    os.makedirs(out_dir, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()},
               os.path.join(out_dir, 'model.pt'))
    with open(os.path.join(out_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab.idx2str, f)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump({
            'd_model': args.d_model, 'n_layers': args.n_layers, 'n_heads': args.n_heads,
            'max_len': args.max_len, 'dropout': args.dropout,
            # training provenance (not needed for inference)
            'source_csv': os.path.basename(args.csv), 'n_sequences': len(dataset),
            'vocab_size': vocab.size, 'min_count': args.min_count,
            'epochs_run': stopped_at, 'best_loss': best_loss, 'seed': args.seed,
        }, f, indent=2)
    print(f'Saved checkpoint to {out_dir}  (best loss {best_loss:.4f})')


if __name__ == '__main__':
    main()
