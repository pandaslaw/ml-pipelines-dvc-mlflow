import argparse

import joblib
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV


def train_model(params_path: str) -> None:
    """Train model and save as a dump."""

    params = yaml.safe_load(open(params_path))
    train_df = pd.read_csv(params["data_split"]["path_to_train_dataset"])
    estimator_name = params["train"]["estimator_name"]
    param_grid = params["train"]["estimators"][estimator_name]["param_grid"]
    cv = params["train"]["cv"]
    target_column = params["featurize"]["target_column"]
    # lr_param_grid = {'lr__C': np.arange(-0.0, 1.0, 0.1),  # [0.01, 0.1, 1.0, 10.0], #
    #                  'lr__penalty': ['l1', 'l2']}
    f1_scorer = make_scorer(f1_score, average='weighted')
    clf = GridSearchCV(
        LogisticRegression(),
        param_grid=param_grid,
        cv=cv,
        verbose=1,
        scoring=f1_scorer
    )
    X_train = train_df.drop(target_column, axis=1).values.astype('float32')
    y_train = train_df.loc[:, target_column].values.astype('int32')
    clf.fit(X_train, y_train)
    print(clf.best_score_)
    path_to_model = params["train"]["path_to_model"]
    joblib.dump(clf, path_to_model)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--paramsFile', dest='paramsFile', required=True)
    args = args_parser.parse_args()
    train_model(args.paramsFile)
