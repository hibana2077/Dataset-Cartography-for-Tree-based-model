import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import yaml
import json
import os
import warnings
from tqdm import tqdm
from rich import print as rprint
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from train_model import log_reg_train, random_forest_train, decision_tree_train, gradient_boosting_train

warnings.filterwarnings("ignore")

EXP_REC = {}

# define arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="path to config file")

# read arguments from the command line
args = parser.parse_args()
CONFIG_PATH = args.config

def decrease_max_iter(index:int) -> int:
    base = 1300
    return max(100, base - index * 300)

def decrease_C(index:int) -> float:
    base = 100
    return max(0.1, base - index * 20)

def decrease_n_estimators(index: int) -> int:
    base = 100  # 可以設定一個基礎數值
    return max(10, base - index * 20)  # 確保 n_estimators 不會小於某個最小值，例如 10

def decrease_max_depth(index: int) -> int:
    base = 50  # 可以設定一個基礎數值
    return max(5, base - index * 10)  # 確保 max_depth 不會小於某個最小值，例如 5

# read config file
with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

# read parameters
DATA_PATH = config['train']['data_path']
TARGET = config['train']["target"]

# read data
train_df = pd.read_csv(DATA_PATH+"train.csv")
test_df = pd.read_csv(DATA_PATH+"test.csv")

# define data preprocessing pipeline
data_preprocess = Pipeline([
    ('label_encoding', LabelEncoder()),
    ('standard_scaling', StandardScaler())
])

# preprocess data
X_train = data_preprocess.fit_transform(train_df.drop(TARGET, axis=1))
y_train = train_df[TARGET]

X_test = data_preprocess.transform(test_df.drop(TARGET, axis=1))
y_test = test_df[TARGET]

y_train = y_train.astype(int)
y_test = y_test.astype(int)

rprint("[bold green]Data Preprocessing Completed![/bold green]")

# save data shapes to experiment record
EXP_REC["train_shape"] = X_train.shape
EXP_REC["test_shape"] = X_test.shape
EXP_REC["target"] = TARGET

# main loop (train model for datasets cartography -> stacking)
try:
    for model_config in config['train']['models']:
        model_name = model_config['name']
        model_params = model_config['params_range']
        rprint("[bold green]Training {} Model...[/bold green]".format(model_name))
        if model_name == "LogisticRegression":
            prob, correct, train_accuracy, test_accuracy, train_confusion_matrix, train_classification_report, test_confusion_matrix, test_classification_report = log_reg_train(X_train, y_train, X_test, y_test, model_params)
        elif model_name == "RandomForest":
            prob, correct, train_accuracy, test_accuracy, train_confusion_matrix, train_classification_report, test_confusion_matrix, test_classification_report = random_forest_train(X_train, y_train, X_test, y_test, model_params)
        elif model_name == "DecisionTree":
            prob, correct, train_accuracy, test_accuracy, train_confusion_matrix, train_classification_report, test_confusion_matrix, test_classification_report = decision_tree_train(X_train, y_train, X_test, y_test, model_params)
        elif model_name == "GradientBoosting":
            prob, correct, train_accuracy, test_accuracy, train_confusion_matrix, train_classification_report, test_confusion_matrix, test_classification_report = gradient_boosting_train(X_train, y_train, X_test, y_test, model_params)
        else:
            raise Exception("Invalid Model Name")
        # save model training results to experiment record
        EXP_REC[model_name] = {}
        EXP_REC[model_name]["train_accuracy"] = train_accuracy
        EXP_REC[model_name]["test_accuracy"] = test_accuracy
        EXP_REC[model_name]["train_confusion_matrix"] = train_confusion_matrix
        EXP_REC[model_name]["train_classification_report"] = train_classification_report
        EXP_REC[model_name]["test_confusion_matrix"] = test_confusion_matrix
        EXP_REC[model_name]["test_classification_report"] = test_classification_report

        rprint("[bold green]Training Completed![/bold green]")
        rprint("[bold green]Making Datasets Cartography...[/bold green]")
        # make datasets cartography
        prob = np.array(prob)
        correct = np.array(correct)
        prob = prob.T
        correct = correct.T
        mean = np.mean(prob, axis=1)
        std = np.std(prob, axis=1)
        varibility = std / mean
        correctness = np.mean(correct, axis=1)
        plt.figure()
        plt.scatter(varibility, mean, c=correctness, cmap='coolwarm')
        plt.colorbar(label = 'correctness')
        plt.xlabel('varibility')
        plt.ylabel('mean')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title('Datasets Cartography')
        plt.savefig(DATA_PATH + model_name + "_cartography.png")
        correctness_min, correctness_max = np.min(correctness), np.max(correctness)
        # pick range
        correctness_range = [[min_i, min_i + 0.3] for min_i in np.arange(correctness_min, 1.2, 0.4)]
        rprint(correctness_range)

        pick = []
        for j in range(len(correctness_range)):
            pick.append([idx for idx,i in enumerate(correctness) if correctness_range[j][0] <= i < correctness_range[j][1]])

        pick_train_x = []
        pick_train_y = []
        for i in pick:
            pick_train_x.append(X_train.iloc[i, :])
            pick_train_y.append(y_train.iloc[i])

except Exception as e:
    rprint("[bold red]Error:[/bold red] {}".format(str(e)))
    # save experiment record
    EXP_REC["error"] = str(e)
    file_name = CONFIG_PATH + CONFIG_PATH.split("/")[-1] + "_exp_rec.json"
    with open(file_name, "w") as file:
        json.dump(EXP_REC, file)
    exit(1)