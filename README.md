# 資料探勘期末作業

## Requirements
+ Conda
+ CUDA 11.8

## Directory Structure
```
datamining_finalproject/
├── test/(self-loading)
│ ├── test_behaviors.tsv
| ├── test_entity_embedding.vec
│ └── test_news.tsv
├── train/(self-loading)
│ ├── train_behaviors.tsv
| ├── train_entity_embedding.vec
│ └── train_news.tsv
├── .gitignore
├── environment.yml
├── README.md
├── train.py
├── evaluate.py
├── model.pth(generate)
└── submission.csv(generate)
```

## Build Steps

### Install environment and dependencies
```
conda env create -f environment.yml
conda activate final
```

### Run 

```
python train.py
python evaluate.py
```

### Fix csv
if csv id is not mesh, you can try
```
python fix/fix.py
```