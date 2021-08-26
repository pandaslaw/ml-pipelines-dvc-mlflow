import argparse

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def data_split(params_path: str) -> None:
    """Split dataset on train and test."""

    params = yaml.safe_load(open(params_path))

    dataset = pd.read_csv(params["featurize"]["path_to_featured_dataset"])
    dataset_train, dataset_test = train_test_split(
        dataset,
        test_size=params["data_split"]["test_size"],
        random_state=params["base"]["random_state"]
    )

    path_train = params["data_split"]["path_to_train_dataset"]
    path_test = params["data_split"]["path_to_test_dataset"]
    dataset_train.to_csv(path_train, index=False)
    dataset_test.to_csv(path_test, index=False)
    
    print(f'Train dataset size: {dataset_train.shape}. Saved to: {path_train}')
    print(f'Test dataset size: {dataset_test.shape}. Saved to: {path_test}')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--paramsFile', dest='paramsFile', required=True)
    args = args_parser.parse_args()
    data_split(args.paramsFile)
