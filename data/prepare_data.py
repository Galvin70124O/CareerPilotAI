"""
Download and clean the ResumeAtlas dataset, then save it as data/resumes.parquet.

Dataset: https://huggingface.co/datasets/ahmedheakl/resume-atlas  (MIT licence)
Paper:   https://arxiv.org/abs/2406.18125

Run from the project folder:
    python data/prepare_data.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Let this script import from the project folder (features/...)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from features.text_cleaning import normalize_text  # noqa: E402

# Pinned to one dataset version, so everyone downloads exactly the same data
DATASET_REVISION = "3f80ca910fa9964890afb7845c09e239d07b9b1d"
DATASET_URL = f"https://huggingface.co/datasets/ahmedheakl/resume-atlas/resolve/{DATASET_REVISION}/train.csv"
OUTPUT_PATH = os.path.join(PROJECT_DIR, "data", "resumes.parquet")

MIN_WORDS = 50                    # shorter "resumes" are mostly empty or broken
NEAR_DUPLICATE_THRESHOLD = 0.9    # cosine similarity above this = near-duplicate

# The dataset authors hid real names by replacing them with "claire". In some resumes this
# went wrong and replaced single LETTERS too: "with" -> "wiclaireh", "the" -> "clairehe".
# These broken words can't be repaired (we don't know which letter was replaced).
PLACEHOLDER = "claire"
MAX_BROKEN_SHARE = 0.05           # drop a resume if more than 5% of its words are broken


def remove_placeholder_junk(text):
    """Return (cleaned_text, share_of_broken_words)."""
    words = text.split()
    broken = [w for w in words if PLACEHOLDER in w and w != PLACEHOLDER]
    kept = [w for w in words if PLACEHOLDER not in w]
    return " ".join(kept), len(broken) / max(len(words), 1)


def remove_near_duplicates(texts):
    """Return a True/False list: True = keep. The first copy of a near-duplicate group is kept."""
    # TF-IDF vectors have length 1, so a dot product between two rows IS their cosine similarity
    vectors = TfidfVectorizer(min_df=2).fit_transform(texts)
    keep = np.ones(len(texts), dtype=bool)
    chunk_size = 1000  # compare 1000 resumes at a time against all others, to save memory
    for start in range(0, len(texts), chunk_size):
        similarity = (vectors[start:start + chunk_size] @ vectors.T).toarray()
        for offset, row in enumerate(similarity):
            i = start + offset
            earlier_similar = np.where(row[:i] > NEAR_DUPLICATE_THRESHOLD)[0]
            # Drop this resume if it is too similar to an earlier resume we kept
            if keep[earlier_similar].any():
                keep[i] = False
    return keep


# --- Label fixes (checked by hand, see README "Label noise") ---------------------------
# 1) Many "Data Science" resumes are school SCIENCE TEACHERS. If a Data Science resume talks
#    about teaching and mentions none of the data tools below, we move it to "Education".
#    The tool list is generous on purpose: a real data person should never be moved.
# (?:...) groups words without "capturing" them; pandas warns about capturing groups
TEACHING_WORDS = r"\b(?:teacher|classroom|grade|lesson|students)\b"
DATA_TOOL_WORDS = (r"\b(?:python|machine learning|pandas|numpy|scikitlearn|tensorflow|keras|data scientist|"
                   r"statistics|sql|r programming|tableau|power bi|powerbi|data analysis|spark)\b")
# 2) "Management" is a job level found in every industry, not a career path (the model got
#    only 26% of them right), so the category is dropped.
DROPPED_CATEGORIES = ["Management"]


def fix_labels(data):
    """Return (fixed_data, moved_mask). moved_mask marks Data Science rows moved to Education."""
    data = data.copy()
    is_data_science = data["Category"] == "Data Science"
    teaches = data["Resume"].str.contains(TEACHING_WORDS)
    uses_data_tools = data["Resume"].str.contains(DATA_TOOL_WORDS)
    moved = is_data_science & teaches & ~uses_data_tools
    data.loc[moved, "Category"] = "Education"
    keep = ~data["Category"].isin(DROPPED_CATEGORIES)
    return data[keep].reset_index(drop=True), moved[keep].reset_index(drop=True)


def main():
    print(f"Downloading {DATASET_URL} ...")
    data = pd.read_csv(DATASET_URL).rename(columns={"Text": "Resume"})
    print(f"Downloaded:                         {len(data)} rows, {data['Category'].nunique()} categories")

    # 1) Empty and very short resumes
    data = data.dropna(subset=["Resume", "Category"])
    data["Resume"] = data["Resume"].map(normalize_text)   # also removes any encoding junk
    data["Category"] = data["Category"].str.strip()

    # Name placeholder junk ("claire", "wiclaireh", ...)
    cleaned = data["Resume"].map(remove_placeholder_junk)
    data["Resume"] = cleaned.str[0]
    broken_share = cleaned.str[1]
    data = data[broken_share <= MAX_BROKEN_SHARE]
    print(f"After dropping resumes with >{MAX_BROKEN_SHARE:.0%} broken words: {len(data)}")

    data = data[data["Resume"].str.split().str.len() >= MIN_WORDS]
    print(f"After dropping resumes < {MIN_WORDS} words:   {len(data)}")

    # 2) The same text with two DIFFERENT categories is confusing for the model: drop all copies
    labels_per_text = data.groupby("Resume")["Category"].transform("nunique")
    data = data[labels_per_text == 1]
    print(f"After dropping conflicting labels:  {len(data)}")

    # 3) Exact duplicates: keep one copy
    data = data.drop_duplicates(subset="Resume").reset_index(drop=True)
    print(f"After dropping exact duplicates:    {len(data)}")

    # 4) Near-duplicates (almost the same text)
    keep = remove_near_duplicates(data["Resume"])
    data = data[keep].reset_index(drop=True)
    print(f"After dropping near-duplicates:     {len(data)}  ({(~keep).sum()} removed, cosine > {NEAR_DUPLICATE_THRESHOLD})")

    # 5) Label fixes: science teachers -> Education, drop "Management"
    data, moved = fix_labels(data)
    print(f"Moved Data Science -> Education:    {moved.sum()} science-teacher resumes without data tools")
    print(f"After dropping {', '.join(DROPPED_CATEGORIES)}:         {len(data)}")

    counts = data["Category"].value_counts()
    print(f"Categories: {len(counts)}  (smallest: {counts.idxmin()} = {counts.min()}, largest: {counts.idxmax()} = {counts.max()})")

    data.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
