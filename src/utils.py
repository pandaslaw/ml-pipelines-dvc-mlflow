import box
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import plot_confusion_matrix


def load_config(config_path: str) -> box.ConfigBox:
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
        config = box.ConfigBox(config)
        return config


# create categorical column based on Parch and SibSp
def get_family_type(row):
    total = row.Parch + row.SibSp + 1
    if total >= 5:
        return 'Large'
    if 2 <= total <= 4:
        return 'Small'
    return 'Single'


# create Age category
def get_age_category(age):
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


def fit_plot_confusion(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    mean, std = clf.cv_results_['mean_test_score'][clf.best_index_], \
                clf.cv_results_['std_test_score'][clf.best_index_]
    print(clf.best_params_)
    disp = plot_confusion_matrix(clf, X_test, y_test, normalize='true')
    disp.figure_.suptitle("Confusion Matrix")
    plt.show()
    return clf.best_estimator_, {"mean": mean, "std": std}
