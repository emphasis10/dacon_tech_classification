# Introduction
This repo is for [Dacon competition](https://dacon.io/competitions/official/235744/overview/description). After competition, all submitted codes will be opened.

# Prerequisite

## Data Preparation
First of all, you should download data from Dacon [Competition site](https://dacon.io/competitions/official/235744/data). You will get 4 files which are `train.csv`, `test.csv`, `sample_submission.csv`, `labels_mapping.csv` and move these files into `{project_root_dir}/resource/data/`. At first, there's no `resource` directory, you may create directory using like `mkdir` command or any other method preferred.

## Package Installation
```shell
$ pip install -r requirements.txt
```

# Execution
To understand basic pipeline, check `toy.py`.
```Shell
$ python toy.py
```
It contains simple CNN as base model. You can customize it simply.

# Results
Following without any changes, prediction results about test data exported in `./results`.

# Competition Records
[Official Leaderboard](https://dacon.io/competitions/official/235744/leaderboard)

Our team name was `SRiracha`, we ranked on 21th place on private dataset. We finally use ensemble model with previous submitted model. Our model scored(MacroF1) `0.78056` on private data.

# Contributor
- Eunsik Lee(@emphasis10)