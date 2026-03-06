"""
Build MIND (Microsoft News) dataset.

Raw data should be placed at:
    data/mind/train/behaviors.tsv
    data/mind/train/news.tsv
    data/mind/valid/behaviors.tsv
    data/mind/valid/news.tsv

Download MIND-Large from https://msnews.github.io/ :
  - MINDlarge_train.zip → extract to data/mind/train/
  - MINDlarge_dev.zip   → extract to data/mind/valid/

Run from the project root:
    python build_dataset/build_dataset_mind.py

Output:
    datasets/MIND/train_sampled_50k.tsv
    datasets/MIND/valid_sampled_50k.tsv
    datasets/MIND/user_dict_mind.pickle
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import pandas as pd
from pathlib import Path
from utils.datasets import EntityDictionary

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ───────────────────────────────────────────────────────────────────
RAW_DIR    = Path("data/mind")
OUTPUT_DIR = Path("datasets/MIND")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load behaviors ───────────────────────────────────────────────────────────
behavior_cols = ['impression_id', 'user_id', 'time', 'history', 'impressions']
df_train_raw = pd.read_csv(RAW_DIR / "train/behaviors.tsv", sep='\t',
                            header=None, names=behavior_cols)
df_valid_raw = pd.read_csv(RAW_DIR / "valid/behaviors.tsv", sep='\t',
                            header=None, names=behavior_cols)

# Keep only users that appear in both splits
common_users = set(df_train_raw['user_id']) & set(df_valid_raw['user_id'])
df_train_beh = df_train_raw[df_train_raw['user_id'].isin(common_users)].copy()
df_valid_beh = df_valid_raw[df_valid_raw['user_id'].isin(common_users)].copy()
print(f"Common users in train & valid: {len(common_users):,}")

# ── Load news and build title map ────────────────────────────────────────────
news_cols = ['news_id', 'category', 'subcategory', 'title', 'abstract',
             'url', 'title_entities', 'abstract_entities']
df_news = pd.concat([
    pd.read_csv(RAW_DIR / "train/news.tsv", sep='\t', header=None, names=news_cols),
    pd.read_csv(RAW_DIR / "valid/news.tsv", sep='\t', header=None, names=news_cols),
]).drop_duplicates(subset='news_id')
news_title_map = dict(zip(df_news['news_id'], df_news['title']))
print(f"Unique news items: {len(news_title_map):,}")

# ── Parse history IDs, map to sampled titles (up to 5) ──────────────────────
def parse_history(s):
    return [] if (pd.isna(s) or s == '') else s.split()

def ids_to_sampled_titles(news_ids, title_map, n=5):
    titles = [title_map[nid] for nid in news_ids if nid in title_map]
    return random.sample(titles, min(n, len(titles))) if titles else []

for df_beh in (df_train_beh, df_valid_beh):
    df_beh['history_ids'] = df_beh['history'].apply(parse_history)

# ── Explode impressions into individual (user, news, label) rows ─────────────
def explode_impressions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['impressions'] = df['impressions'].apply(
        lambda s: [] if pd.isna(s) else [
            (p.split('-')[0], int(p.split('-')[1]))
            for p in s.split() if '-' in p
        ]
    )
    df = df.explode('impressions').dropna(subset=['impressions'])
    df['news_id'] = df['impressions'].apply(lambda x: x[0])
    df['Label']   = df['impressions'].apply(lambda x: x[1])
    df['News_Title'] = df['news_id'].map(news_title_map)
    return df.drop(columns=['impressions', 'history', 'impression_id', 'time',
                             'news_id']).reset_index(drop=True)

df_train = explode_impressions(df_train_beh)
df_valid = explode_impressions(df_valid_beh)

# ── Filter: users must have history length >= 7 in both splits ───────────────
train_long_hist = set(df_train_beh[df_train_beh['history_ids'].apply(len) >= 7]['user_id'])
valid_long_hist = set(df_valid_beh[df_valid_beh['history_ids'].apply(len) >= 7]['user_id'])
users_with_history = train_long_hist & valid_long_hist
df_train = df_train[df_train['user_id'].isin(users_with_history)].reset_index(drop=True)
df_valid = df_valid[df_valid['user_id'].isin(users_with_history)].reset_index(drop=True)

# ── Filter: users with >= 4 positive and >= 20 total interactions (train) ────
user_stats = df_train.groupby('user_id')['Label'].agg(total='count', positive='sum')
qualified = user_stats[(user_stats['positive'] >= 4) & (user_stats['total'] >= 20)].index
df_train = df_train[df_train['user_id'].isin(qualified)].reset_index(drop=True)
df_valid = df_valid[df_valid['user_id'].isin(qualified)].reset_index(drop=True)
print(f"After quality filtering: {df_train['user_id'].nunique():,} users")

# ── Cap interactions per user (train: 60, valid: 20) ─────────────────────────
def cap_interactions(df: pd.DataFrame, max_total: int,
                     max_pos_ratio: float = 0.5) -> pd.DataFrame:
    parts = []
    for _, user_df in df.groupby('user_id'):
        if len(user_df) <= max_total:
            parts.append(user_df)
            continue
        pos_df = user_df[user_df['Label'] == 1]
        neg_df = user_df[user_df['Label'] == 0]
        n_pos = min(len(pos_df), int(max_total * max_pos_ratio))
        n_neg = max_total - n_pos
        parts.append(pd.concat([
            pos_df.sample(n_pos, random_state=SEED) if n_pos < len(pos_df) else pos_df,
            neg_df.sample(n_neg, random_state=SEED) if n_neg < len(neg_df) else neg_df,
        ]))
    return pd.concat(parts).reset_index(drop=True)

df_train = cap_interactions(df_train, max_total=60)
df_valid = cap_interactions(df_valid, max_total=20)

# ── Per-row independent history sampling (data augmentation) ────────────────
# Each row samples independently from the full session history, so different
# candidate news items in the same session see different history subsets.
# Reset seed here to match the notebook's random.seed(42) before sample_history.
random.seed(SEED)
df_train['History_Interact_Title'] = df_train['history_ids'].apply(
    lambda ids: ids_to_sampled_titles(ids, news_title_map)
)
df_valid['History_Interact_Title'] = df_valid['history_ids'].apply(
    lambda ids: ids_to_sampled_titles(ids, news_title_map)
)
df_train = df_train.drop(columns=['history_ids'])
df_valid = df_valid.drop(columns=['history_ids'])

# ── Filter: drop rows where joined history title string > 512 chars ──────────
def joined_len(titles):
    return len(' '.join(titles)) if titles else 0

df_train = df_train[df_train['History_Interact_Title'].apply(joined_len) <= 512].reset_index(drop=True)
df_valid = df_valid[df_valid['History_Interact_Title'].apply(joined_len) <= 512].reset_index(drop=True)

# ── Filter: drop users with < 10 valid interactions ─────────────────────────
# Applied after the 512-char filter (matching the notebook order).
valid_user_counts = df_valid.groupby('user_id').size()
enough_valid = valid_user_counts[valid_user_counts >= 10].index
df_train = df_train[df_train['user_id'].isin(enough_valid)].reset_index(drop=True)
df_valid = df_valid[df_valid['user_id'].isin(enough_valid)].reset_index(drop=True)

# ── Sample up to 50k users ───────────────────────────────────────────────────
# Reset seed to match the notebook's random.seed(42) before 50k sampling.
random.seed(SEED)
all_users = list(df_train['user_id'].unique())
sampled_users = random.sample(all_users, min(50_000, len(all_users)))
df_train = df_train[df_train['user_id'].isin(sampled_users)].reset_index(drop=True)
df_valid = df_valid[df_valid['user_id'].isin(sampled_users)].reset_index(drop=True)
print(f"Final: train {len(df_train):,} rows, valid {len(df_valid):,} rows, "
      f"{len(sampled_users):,} users")


# ── Build user dictionary ─────────────────────────────────────────────────────
user_dict = EntityDictionary()
for uid in np.random.permutation(sampled_users):
    user_dict.add_entity(uid)

user_dict_path = OUTPUT_DIR / "user_dict_mind.pickle"
user_dict.save(str(user_dict_path))
print(f"User dict saved → {user_dict_path}")

# ── Rename columns and save ──────────────────────────────────────────────────
df_train = df_train.rename(columns={"user_id": "UserID"})
df_valid = df_valid.rename(columns={"user_id": "UserID"})

train_path = OUTPUT_DIR / "train_sampled_50k.tsv"
valid_path = OUTPUT_DIR / "valid_sampled_50k.tsv"
df_train.to_csv(train_path, sep="\t", index=False)
df_valid.to_csv(valid_path, sep="\t", index=False)
print(f"Saved  {train_path}")
print(f"Saved  {valid_path}")
