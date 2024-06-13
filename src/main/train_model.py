from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# define model training function
def log_reg_train(X_train, y_train, X_test, y_test, model_params):
    prob = []
    correct = []
    train_accuracy = {}
    train_confusion_matrix = {}
    train_classification_report = {}
    test_accuracy = {}
    test_confusion_matrix = {}
    test_classification_report = {}
    # model_params = {'C': [0.1, 1, 10, 100, 1000], max_iter: [100, 200, 300, 400, 500]}
    for c, max_iter in tqdm(model_params):
        model = LogisticRegression(C=c, max_iter=max_iter)
        model.fit(X_train, y_train)
        y_pred_train = model.predict_proba(X_train)
        prob.append(y_pred_train[np.arange(len(y_train)), y_train])
        correct.append(y_train == np.argmax(y_pred_train, axis=1))
        y_pred = model.predict(X_test)
        train_accuracy[f"C={c}, max_iter={max_iter}"] = accuracy_score(y_train, np.argmax(y_pred_train, axis=1))
        train_confusion_matrix[f"C={c}, max_iter={max_iter}"] = confusion_matrix(y_train, np.argmax(y_pred_train, axis=1))
        train_classification_report[f"C={c}, max_iter={max_iter}"] = classification_report(y_train, np.argmax(y_pred_train, axis=1))
        test_accuracy[f"C={c}, max_iter={max_iter}"] = accuracy_score(y_test, y_pred)
        test_confusion_matrix[f"C={c}, max_iter={max_iter}"] = confusion_matrix(y_test, y_pred)
        test_classification_report[f"C={c}, max_iter={max_iter}"] = classification_report(y_test, y_pred)
    return prob, correct, train_accuracy, test_accuracy, train_confusion_matrix, train_classification_report, test_confusion_matrix, test_classification_report

def random_forest_train(X_train, y_train, X_test, y_test, model_params):
    prob = []
    correct = []
    train_accuracy = {}
    train_confusion_matrix = {}
    train_classification_report = {}
    test_accuracy = {}
    test_confusion_matrix = {}
    test_classification_report = {}
    # model_params = {'n_estimators': [50, 100, 150, 200], 'max_depth': [10, 20, 30, 40, 50]}
    for n_estimators, max_depth in tqdm(model_params):
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        y_pred_train = model.predict_proba(X_train)
        prob.append(y_pred_train[np.arange(len(y_train)), y_train])
        correct.append(y_train == np.argmax(y_pred_train, axis=1))
        y_pred = model.predict(X_test)
        train_accuracy[f"n_estimators={n_estimators}, max_depth={max_depth}"] = accuracy_score(y_train, np.argmax(y_pred_train, axis=1))
        train_confusion_matrix[f"n_estimators={n_estimators}, max_depth={max_depth}"] = confusion_matrix(y_train, np.argmax(y_pred_train, axis=1))
        train_classification_report[f"n_estimators={n_estimators}, max_depth={max_depth}"] = classification_report(y_train, np.argmax(y_pred_train, axis=1))
        test_accuracy[f"n_estimators={n_estimators}, max_depth={max_depth}"] = accuracy_score(y_test, y_pred)
        test_confusion_matrix[f"n_estimators={n_estimators}, max_depth={max_depth}"] = confusion_matrix(y_test, y_pred)
        test_classification_report[f"n_estimators={n_estimators}, max_depth={max_depth}"] = classification_report(y_test, y_pred)
    return prob, correct, train_accuracy, test_accuracy, train_confusion_matrix, train_classification_report, test_confusion_matrix, test_classification_report

def decision_tree_train(X_train, y_train, X_test, y_test, model_params):
    prob = []
    correct = []
    train_accuracy = {}
    train_confusion_matrix = {}
    train_classification_report = {}
    test_accuracy = {}
    test_confusion_matrix = {}
    test_classification_report = {}
    # model_params = {'max_depth': [10, 20, 30, 40, 50]}
    for max_depth in tqdm(model_params):
        model = DecisionTreeClassifier(max_depth=max_depth)
        model.fit(X_train, y_train)
        y_pred_train = model.predict_proba(X_train)
        prob.append(y_pred_train[np.arange(len(y_train)), y_train])
        correct.append(y_train == np.argmax(y_pred_train, axis=1))
        y_pred = model.predict(X_test)
        train_accuracy[f"max_depth={max_depth}"] = accuracy_score(y_train, np.argmax(y_pred_train, axis=1))
        train_confusion_matrix[f"max_depth={max_depth}"] = confusion_matrix(y_train, np.argmax(y_pred_train, axis=1))
        train_classification_report[f"max_depth={max_depth}"] = classification_report(y_train, np.argmax(y_pred_train, axis=1))
        test_accuracy[f"max_depth={max_depth}"] = accuracy_score(y_test, y_pred)
        test_confusion_matrix[f"max_depth={max_depth}"] = confusion_matrix(y_test, y_pred)
        test_classification_report[f"max_depth={max_depth}"] = classification_report(y_test, y_pred)
    return prob, correct, train_accuracy, test_accuracy, train_confusion_matrix, train_classification_report, test_confusion_matrix, test_classification_report

def gradient_boosting_train(X_train, y_train, X_test, y_test, model_params):
    prob = []
    correct = []
    train_accuracy = {}
    train_confusion_matrix = {}
    train_classification_report = {}
    test_accuracy = {}
    test_confusion_matrix = {}
    test_classification_report = {}
    # model_params = {'n_estimators': [50, 100, 150, 200], 'max_depth': [10, 20, 30, 40, 50]}
    for n_estimators, max_depth in tqdm(model_params):
        model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        y_pred_train = model.predict_proba(X_train)
        prob.append(y_pred_train[np.arange(len(y_train)), y_train])
        correct.append(y_train == np.argmax(y_pred_train, axis=1))
        y_pred = model.predict(X_test)
        train_accuracy[f"n_estimators={n_estimators}, max_depth={max_depth}"] = accuracy_score(y_train, np.argmax(y_pred_train, axis=1))
        train_confusion_matrix[f"n_estimators={n_estimators}, max_depth={max_depth}"] = confusion_matrix(y_train, np.argmax(y_pred_train, axis=1))
        train_classification_report[f"n_estimators={n_estimators}, max_depth={max_depth}"] = classification_report(y_train, np.argmax(y_pred_train, axis=1))
        test_accuracy[f"n_estimators={n_estimators}, max_depth={max_depth}"] = accuracy_score(y_test, y_pred)
        test_confusion_matrix[f"n_estimators={n_estimators}, max_depth={max_depth}"] = confusion_matrix(y_test, y_pred)
        test_classification_report[f"n_estimators={n_estimators}, max_depth={max_depth}"] = classification_report(y_test, y_pred)
    return prob, correct, train_accuracy, test_accuracy, train_confusion_matrix, train_classification_report, test_confusion_matrix, test_classification_report
