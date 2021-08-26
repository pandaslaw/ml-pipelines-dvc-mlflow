import argparse

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler


def get_family_type(row: pd.Series) -> str:
    """Create categorical column based on Parch and SibSp"""

    total = row.Parch + row.SibSp + 1
    if total >= 5:
        return 'Large'
    if 2 <= total <= 4:
        return 'Small'
    return 'Single'


def get_age_category(age: pd.Series) -> int:
    """Create Age category."""

    if age < 0:
        return -1
    if 0 <= age < 4:
        return 0
    if 4 <= age < 16:
        return 1
    if 16 <= age < 36:
        return 2
    if 36 <= age < 52:
        return 3
    if 52 <= age < 64:
        return 4
    if age >= 64:
        return 5


def extract_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Extract and preprocess features."""

    dataset.set_index('PassengerId', inplace=True)
    dataset.loc[(dataset.Embarked.isnull()) & (dataset.Survived == 1), 'Embarked'] = 'C'
    dataset.loc[(dataset.Embarked.isnull()) & (dataset.Survived == 0), 'Embarked'] = 'S'
    dataset.loc[(dataset.Embarked.isnull()) & (dataset.Survived.isnull()), 'Embarked'] = 'S'

    dataset.loc[dataset.Cabin.isnull(), 'Cabin'] = "U"
    dataset.Fare = dataset.Fare.fillna(dataset.Fare.mean())

    # extract passenger's title from Name column
    second_part = dataset.Name.str.split(',').str[1]
    dataset['Title'] = pd.DataFrame(second_part).Name.str.split('\.').str[0]
    dataset['Title'] = dataset['Title'].str.strip()
    # let's simplify title
    title_dict = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Dona": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",  # Master is a title for an underage male
        "Lady": "Royalty"
    }
    dataset['Title'] = dataset.Title.map(title_dict)
    age_median_by_name = dataset.groupby('Title')['Age'].median()
    dataset.Age = dataset.apply(lambda row: age_median_by_name[row['Title']] if np.isnan(row['Age']) else row['Age'],
                                axis=1)

    # extract cabin type from Cabin
    cabins_dict = dict((x, i) for i, x in enumerate(set(dataset.Cabin.str[0])))
    dataset['Cabin_type'] = dataset.Cabin.apply(lambda cabin: cabins_dict.get(cabin[0], -1))
    dataset.Parch.value_counts()
    dataset['Family_type'] = dataset.apply(get_family_type, axis=1)
    dataset['Age_type'] = dataset.Age.apply(lambda age: get_age_category(age))

    # Normalize numeric predictors
    numeric_features = ['Age', 'Fare']
    scaler = StandardScaler()
    scaler.fit(dataset[numeric_features])
    dataset[numeric_features] = scaler.transform(dataset[numeric_features])

    # Encode categorical features
    categorical_features = ['Pclass', 'Sex', 'Cabin_type', 'Embarked', 'Title', 'Family_type']  # , 'Age_type'
    dataset = pd.get_dummies(dataset, columns=categorical_features, drop_first=True)
    dataset.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
    return dataset


def featurize(params_path: str) -> None:
    """Prepare a dataset for future modelling."""

    params = yaml.safe_load(open(params_path))
    dataset = pd.read_csv(params["data"]["path_to_train_dataset"])
    featured_dataset = extract_features(dataset)
    featured_path = params["featurize"]["path_to_featured_dataset"]
    featured_dataset.to_csv(featured_path)
    print(f'Features saved to: {featured_path}')
    print(f'Features shape: {featured_dataset.shape}')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--paramsFile', dest='paramsFile', required=True)
    args = args_parser.parse_args()
    featurize(args.paramsFile)
