# ML pipeline automation and reproducibility

This project demonstrates data versioning, experiment tracking and artifacts logging capabilities using DVC and MLflow.

## Setup local environment

Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```
Install python libraries
```bash
pip install -r requirements.txt
```

## DVC: Automate experiment pipeline
### 1. Install DVC
```bash
pip install dvc
```

### 2. Initialize DVC
Note that in cloned project DVC can be already initialized.
```bash
dvc init
```
Commit the initialization of DVC
```bash
git commit -m "Initialize DVC"
``` 

### 3. Add remote storage for DVC
It can be a local folder (local data backup) or remote storage, e.g. Amazon S3, SSH, Google Drive, Azure Blob Storage or HDFS.
```bash
dvc remote add -d myremote /tmp/dvcstore
git commit .dvc/config -m "Configure local remote"
dvc push
```

### 4. Create pipeline
Use ```dvc run``` to create stages. On executing the commands below ```dvc.yaml``` file will be created.

```bash
dvc run -n featurize \
        -o data/processed/featured_titanic.csv \
        python src/featurize.py --paramsFile=params.yaml
```
```bash
dvc run -n data_split \
        -p data_split.test_size \
        -o data/processed/train_titanic.csv \
        -o data/processed/test_titanic.csv \
        python src/data_split.py --paramsFile=params.yaml
```
```bash
dvc run -n train \
        -p train.cv \
        -o models/model.joblib \
        python src/train.py --paramsFile=params.yaml
```
```bash
dvc run -n evaluate \
        -d models/model.joblib 
        python -m src.utils src/evaluate.py --paramsFile=params.yaml
```
After ```dvc.yaml``` file creation one can easily reproduce a pipeline by running
```bash
dvc repro
```
Note: stages without changes in the parameters won't rerun on executing ```dvc repro```

## MLflow

## Airflow

## Docker


## References
1. [DVC tutorial](https://dvc.org/doc/tutorial)
2. [MLflow tutorial](https://www.mlflow.org/docs/latest/quickstart.html)
3. [Plot a Confusion Matrix](https://www.kaggle.com/grfiv4/plot-a-confusion-matrix) 