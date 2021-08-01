import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.utils import get_family_type, get_age_category, fit_plot_confusion, load_config


def train_model(config_path: str):
    config = load_config(config_path)
    data = pd.read_csv(config.data.path_train).set_index('PassengerId')
    test_data = pd.read_csv(config.data.path_test).set_index('PassengerId')

    print("The percentage of missing values for each feature (train set):")
    print(data.isna().sum(axis=0) / data.shape[0] * 100)
    print("\nThe percentage of missing values for each feature (test set):")
    print(test_data.isna().sum(axis=0) / test_data.shape[0] * 100)

    # concat train and test dataframes for simpler preprocessing
    all_data = pd.concat([data, test_data], sort=False, ignore_index=False)
    indexes = all_data[(all_data.Embarked.isnull())].index
    all_data.loc[(all_data.Embarked.isnull()) & (all_data.Survived == 1), 'Embarked'] = 'C'
    all_data.loc[(all_data.Embarked.isnull()) & (all_data.Survived == 0), 'Embarked'] = 'S'
    all_data.loc[(all_data.Embarked.isnull()) & (all_data.Survived.isnull()), 'Embarked'] = 'S'

    all_data.loc[all_data.Cabin.isnull(), 'Cabin'] = "U"
    all_data.Fare = all_data.Fare.fillna(all_data.Fare.mean())
    # extract passenger's title from Name column
    second_part = all_data.Name.str.split(',').str[1]
    all_data['Title'] = pd.DataFrame(second_part).Name.str.split('\.').str[0]
    all_data['Title'] = all_data['Title'].str.strip()
    titles = all_data.Title.value_counts()

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
    all_data['Title'] = all_data.Title.map(title_dict)
    print(all_data['Title'].value_counts())
    age_median_by_name = all_data.groupby('Title')['Age'].median()
    print("Median age within every title:")
    print(age_median_by_name)
    all_data.Age = all_data.apply(lambda row: age_median_by_name[row['Title']] if np.isnan(row['Age']) else row['Age'],
                                  axis=1)

    y = data['Survived']
    all_data.drop('Survived', axis=1, inplace=True)

    # extract cabin type from Cabin
    cabins_dict = dict((x, i) for i, x in enumerate(set(all_data.Cabin.str[0])))
    all_data['Cabin_type'] = all_data.Cabin.apply(lambda cabin: cabins_dict.get(cabin[0], -1))
    all_data.Parch.value_counts()

    all_data['Family_type'] = all_data.apply(get_family_type, axis=1)

    all_data['Age_type'] = all_data.Age.apply(lambda age: get_age_category(age))

    # split data back into train and test sets
    data = all_data.head(data.shape[0]).copy()
    test_data = all_data.tail(test_data.shape[0]).copy()

    numeric_features = ['Age', 'Fare']
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    categorical_features = ['Pclass', 'Sex', 'Cabin_type', 'Embarked', 'Title', 'Family_type']  # , 'Age_type'
    categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    knn_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', KNeighborsClassifier(n_neighbors=5))])
    scores = cross_val_score(knn_pipeline, data, y, scoring='accuracy', cv=10)
    print("Accuracy using k-neighbors: {:.4f}/{:.4f}".format(scores.mean(), scores.std()))

    lr_l2_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', LogisticRegression(random_state=1))])
    scores = cross_val_score(lr_l2_pipeline, data, y, scoring='accuracy', cv=10)
    print("Accuracy using logistic regression (L2): {:.4f}/{:.4f}".format(scores.mean(), scores.std()))

    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.4, shuffle=True, random_state=4)

    knn_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', KNeighborsClassifier(n_neighbors=5))])
    clf = knn_pipeline.fit(X_train, y_train)
    print(accuracy_score(y_true=y_val, y_pred=clf.predict(X_val)))
    disp = plot_confusion_matrix(clf, X_val, y_val, cmap=plt.cm.Blues, normalize=None)

    # [1, 2, 3, 5, 30, 100]
    knn_param_grid = {'knn__n_neighbors': list(range(1, 21)), 'knn__weights': ['uniform', 'distance']}

    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    n_splits = 3
    pipeline = GridSearchCV(
        Pipeline([
            ('preprocessor', preprocessor),
            ('knn', KNeighborsClassifier())
        ]),
        knn_param_grid,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    )

    pipeline.fit(X_train, y_train)
    results = pipeline.cv_results_
    knn_best_clf, knn_stats = fit_plot_confusion(pipeline, X_train, y_train, X_val, y_val)
    print(knn_stats)
    y_pred = knn_best_clf.predict(X_val)
    accuracy_score(y_pred=y_pred, y_true=y_val)

    lr_param_grid = {'lr__C': np.arange(-0.0, 1.0, 0.1),  # [0.01, 0.1, 1.0, 10.0], #
                     'lr__penalty': ['l1', 'l2']}
    n_splits = 3
    pipeline = GridSearchCV(
        Pipeline([
            ('preprocessor', preprocessor),
            ('lr', LogisticRegression(random_state=42, max_iter=1000, solver='saga', n_jobs=-1))
        ]),
        lr_param_grid,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
        scoring='accuracy'
    )
    pipeline.fit(X_train, y_train)
    results = pipeline.cv_results_
    lr_best_clf, lr_stats = fit_plot_confusion(pipeline, X_train, y_train, X_val, y_val)
    print(lr_stats)
    y_pred = lr_best_clf.predict(X_val)
    accuracy_score(y_pred=y_pred, y_true=y_val)

    lr_param_grid = {'C': np.arange(0.0, 1.1, 0.1),  # [0.01, 0.1, 1.0, 10.0], #
                     'penalty': ['l1', 'l2']}

    n_splits = 5
    min_features_to_select = 1

    clf_feature_selection = RandomForestClassifier(n_estimators=30, random_state=42, class_weight="balanced")
    selector = SelectFromModel(clf_feature_selection)
    # selector = RFECV(estimator=clf_feature_selection, step=1, cv=5,
    #                  min_features_to_select=min_features_to_select, scoring='accuracy')
    # selector = SequentialFeatureSelector(estimator=clf_feature_selection, scoring='accuracy')
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000, solver='saga', n_jobs=-1),
        param_grid=lr_param_grid,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
        scoring='accuracy'
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('grid_search', grid_search)
    ])
    pipeline_feature_selection = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', selector),
        ('grid_search', grid_search)
    ])
    print('Cross val score: {:f}'.format(cross_val_score(pipeline, data, y, scoring='accuracy', cv=5).mean()))
    print('Cross val score with feature selection: {:f}'.format(cross_val_score(pipeline_feature_selection, data, y,
                                                                                scoring='accuracy', cv=5).mean()))

    pipeline.fit(data, y)
    best_model = pipeline['grid_search'].best_estimator_
    print('Best score: {}'.format(pipeline['grid_search'].best_score_))
    print('Best parameters: {}'.format(pipeline['grid_search'].best_params_))
    print("Create dump for best model.")

    model_name = config.base.model.model_name
    models_folder = config.base.model.models_folder
    joblib.dump(best_model, os.path.join(models_folder, model_name))

    test_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', best_model)])
    clf = test_pipeline.fit(data, y)
    predictions = clf.predict(test_data)
    submission = pd.DataFrame({
        "PassengerId": test_data.index,
        "Survived": predictions
    })
    # submission.to_csv(os.path.join(PATH, 'titanic-submission.csv'), index=False)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    train_model(config_path=args.config)
