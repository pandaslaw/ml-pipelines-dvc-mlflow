import argparse
import json

import joblib
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

from src.utils import plot_confusion_matrix


def evaluate_model(params_path: str) -> None:
    """Evaluate model on test dataset and save prediction and metrics."""

    params = yaml.safe_load(open(params_path))
    path_to_model = params["train"]["path_to_model"]
    model = joblib.load(path_to_model)
    test_df = pd.read_csv(params["data_split"]["path_to_test_dataset"])
    target_column = params["featurize"]["target_column"]

    X_test = test_df.drop(target_column, axis=1).values.astype('float32')
    y_test = test_df.loc[:, target_column].values.astype('int32')
    prediction = model.predict(X_test)
    f1_score_value = f1_score(y_true=y_test, y_pred=prediction, average='macro')

    evaluation_result_file = params["evaluate"]["path_to_evaluation_result"]
    json.dump(obj={'f1_score': f1_score_value}, fp=open(evaluation_result_file, 'w'))
    print(f'F1 metrics file saved to: {evaluation_result_file}')

    path_to_confusion_matrix_png = params["evaluate"]["path_to_confusion_matrix_png"]
    matrix = confusion_matrix(prediction, y_test)
    plt_matrix = plot_confusion_matrix(cm=matrix, target_names=['Survived', 'Deceased'], normalize=False)
    plt.savefig(path_to_confusion_matrix_png, bbox_inches="tight")
    print(f'Confusion matrix saved to: {path_to_confusion_matrix_png}')

    path_to_prediction_result = params["evaluate"]["path_to_prediction_result"]
    df = pd.DataFrame({'actual': y_test, 'predicted': prediction})
    df.to_csv(path_to_prediction_result, index=False)
    print(f'Passenger survival actual vs predicted saved to: {path_to_prediction_result}')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--paramsFile', dest='paramsFile', required=True)
    args = args_parser.parse_args()
    evaluate_model(args.paramsFile)
