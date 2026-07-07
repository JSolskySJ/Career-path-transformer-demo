"""Build artifacts/title_flags.json — the SJ / taxonomy-level lookup for
predicted job titles.

Sources (the same raw mongo exports the dataset generator matches against, see
CareerPathTransformer.scala in datawarehouse-ai-datasets):

  jobsSkillsTaxonomy/superTitle        -> L3 (canonical SuperTitles)
  jobsSkillsTaxonomy/jobTitleSynonym   -> L4 (synonyms of an L3)
  jobsSkillsTaxonomy/jobTitleVariation -> L5 (variations of an L3)
  jobTitlesV2                          -> SJ recommendable titles (is_sj_title)

Titles are keyed by the ETL's taxonomy normalisation (strip parenthesised
text and non-ASCII, lowercase, non-alphanumeric -> space, collapse). A title
matching several layers keeps the highest (L3 > L4 > L5), matching the ETL's
priority. Output:

    {"tax": {"<normalised title>": "L3"|"L4"|"L5", ...},
     "sj":  ["<normalised title>", ...]}

Usage:
    python scripts/build_title_flags.py [--env test-prod-sj] [--ai-conf PATH]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo import config
from demo.title_flags import FLAGS_JSON, normalise_title


def read_names(ds_path, s3, column):
    """One raw-level-data parquet dir -> list of ACTIVE names (status column
    honoured when present)."""
    import pyarrow.dataset as pads
    ds = pads.dataset(ds_path, filesystem=s3, format='parquet')
    cols = [column] + (['status'] if 'status' in ds.schema.names else [])
    t = ds.to_table(columns=cols)
    df = t.to_pandas()
    if 'status' in df.columns:
        df = df[df['status'] == 'ACTIVE']
    return df[column].dropna().tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', default='test-prod-sj')
    parser.add_argument('--ai-conf', default=None)
    args = parser.parse_args()

    import pyarrow.fs as pafs
    from pyhocon import ConfigFactory

    partner = args.env.split('-')[-1]
    conf_path = args.ai_conf or os.path.abspath(os.path.join(
        config.DEMO_ROOT, '..', 'datawarehouse-configurations', partner, 'ai', 'ai.conf'))
    conf = ConfigFactory.parse_file(conf_path)
    io = f'{args.env}.io'
    base = conf.get_string(f'{io}.amazonInput').replace('s3a://', '')
    s3 = pafs.S3FileSystem(access_key=conf.get_string(f'{io}.amazonAccessKey'),
                           secret_key=conf.get_string(f'{io}.amazonSecretKey'),
                           region=conf.get_string(f'{io}.amazonRegion'))
    raw = f'{base}/mongo-gw/raw-level-data'

    tax = {}
    # L5 first so higher layers overwrite on collisions (ETL priority L3 > L4 > L5).
    for level, path, col in (
        ('L5', f'{raw}/jobsSkillsTaxonomy/jobTitleVariation/parquet', 'name'),
        ('L4', f'{raw}/jobsSkillsTaxonomy/jobTitleSynonym/parquet', 'name'),
        ('L3', f'{raw}/jobsSkillsTaxonomy/superTitle/parquet', 'name'),
    ):
        names = read_names(path, s3, col)
        n = 0
        for name in names:
            key = normalise_title(name)
            if key:
                tax[key] = level
                n += 1
        print(f'{level}: {n:,} titles ({path.rsplit("/", 2)[-2]})')

    sj = set()
    for name in read_names(f'{raw}/jobTitlesV2/parquet', s3, 'title'):
        key = normalise_title(name)
        if key:
            sj.add(key)
    print(f'SJ: {len(sj):,} recommendable titles (jobTitlesV2)')

    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    with open(FLAGS_JSON, 'w') as f:
        json.dump({'tax': tax, 'sj': sorted(sj)}, f)
    counts = {}
    for v in tax.values():
        counts[v] = counts.get(v, 0) + 1
    print(f'Wrote {FLAGS_JSON}  tax={counts}  sj={len(sj):,}')


if __name__ == '__main__':
    main()
