# Dataset

`resumes.parquet` is a cleaned version of **ResumeAtlas**:

- Source: https://huggingface.co/datasets/ahmedheakl/resume-atlas (revision `3f80ca9`)
- Paper: Heakl et al., "ResumeAtlas: Revisiting Resume Classification with Large-Scale Datasets and Large Language Models", 2024. https://arxiv.org/abs/2406.18125
- Code: https://github.com/noran-mohamed/Resume-Classification-Dataset
- Licence: MIT (as stated on the dataset card). Credit goes to the original authors.

`prepare_data.py` rebuilds `resumes.parquet` from the source. It removes very short resumes,
texts with conflicting labels, exact and near-duplicates (cosine > 0.9), and the "claire"
name-placeholder junk.
