import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

from utils.pandas_utils import group_categoricals_tail
import utils.pimpmatplotlib as pm

# Config
CONFIG_FOLDER = os.path.join("config")
CONFIG_FILE = "features.csv"
DATA_FOLDER = os.path.join("..", "sample_data")
DATA_FILE = "dataset.csv"
MODELS_FOLDER = os.path.join("models")
RESULTS_FOLDER = os.path.join("results")

MODEL_TYPE = "lightgbm"
TAG = "lgb"

PREDICTIONS_FILE = "_".join([MODEL_TYPE, TAG, "pred.csv"])

SAVE = True

# lightgbm params
# use an optimisation method to find the best params
SEED = 17
LGB_PARAMS = {
    "objective": "binary",
    "eval_metric": "logloss",
    "seed": SEED,
    "verbose": 0,
    "max_depth": 8,
    "num_leaves": 22,
    "min_data_in_leaf": 500,
    "colsample_bytree": 0.75,
    "subsample": 0.75,
    "learning_rate": 0.1
}


if __name__ == "__main__":

    pmp = pm.PimpPlot(save=SAVE, folder=os.path.join(RESULTS_FOLDER, "plots"))

    # Load config
    source_config = os.path.join(CONFIG_FOLDER, CONFIG_FILE)
    features = pd.read_csv(source_config, keep_default_na=False, na_values=[""])
    index = features.loc[features[MODEL_TYPE] == "index", "column"].tolist()
    predictors = features.loc[features[MODEL_TYPE] == "predictor", "column"].tolist()
    labels = features.loc[features[MODEL_TYPE] == "label", "column"].tolist()
    categorical = features.loc[(features["categorical"] == 1) & (features["column"].isin(predictors)), "column"].tolist()

    # Load data
    source_file = os.path.join(DATA_FOLDER, DATA_FILE)
    data = pd.read_csv(source_file, usecols=index+labels+predictors+["SET"],
                    sep=";", decimal=".", encoding="latin1",
                    keep_default_na = False, na_values = [""])

    # Preprocessing lightgbm
    group_categoricals_tail(data, categorical)

    # Label encoder categorical variables
    label_encoding = {}
    for col in categorical:
        unique_values = data[col].unique().tolist()
        label_encoding[col] = LabelEncoder()
        label_encoding[col].fit(sorted(unique_values))
        data[col] = label_encoding[col].transform(data[col].values)

    # Split the dataset
    indexes = {"train": None, "valid": None, "test": None}
    for set_name in indexes.keys():
        indexes[set_name] = np.where(data["SET"] == set_name)[0]

    # Get only relevant features
    lgb_features = [x for x in sorted(data.columns.tolist()) if x not in labels + index + ["SET"]]

    d = {}
    for set_name, set_indexes in indexes.items():
        if set_name == "test":
            d[set_name] = data.loc[set_indexes, lgb_features].values
        else:
            d[set_name] = lgb.Dataset(data.loc[set_indexes, lgb_features], 
                                        feature_name=lgb_features, 
                                        categorical_feature=categorical, 
                                        free_raw_data=False)

    predictions = {}
    for label in labels:
        print("----------------------------", end="\n")
        print(label, end="\n")
        print("----------------------------", end="\n\n")
        
        print("Creating the Dataset...")
        for set_name, set_indexes in indexes.items():
            if set_name == "test":
                y_test = data.loc[set_indexes, label].values
            else:
                d[set_name].set_label(data.loc[set_indexes, label].values)
        print("Done!", end="\n\n")

        print("Training lightgbm...")
        bst = lgb.train(params=LGB_PARAMS, 
                        train_set=d["train"],
                        num_boost_round=3000, 
                        valid_sets=[d["valid"]],
                        early_stopping_rounds=5)
        print("Done!", end="\n\n")

        print("Predictions and plots LGB...")
        pred_label = "{0}_PRED_{1}".format(label, TAG.upper())
        predictions[pred_label] = bst.predict(d["test"])
        
        print("Logloss: {}".format(log_loss(y_test, predictions[pred_label])), end="\n\n")

        print(predictions[pred_label])

        threshold_preds = 0

        auc_score = roc_auc_score(y_test, predictions[pred_label])
        accuracy = accuracy_score(d["test"].get_label(), threshold_preds)
        precision = precision_score(d["test"].get_label(), threshold_preds)
        recall = recall_score(d["test"].get_label(), threshold_preds)
        pr_auc = average_precision_score(d["test"].get_label(), predictions[pred_label])
        f2_score = fbeta_score(d["test"].get_label(), threshold_preds, beta=2)
        
        print(f"AUC {label}: {auc_score:.4f}")
        print(f"Accuracy {label}: {accuracy:.4f}")
        print(f"Precision {label}: {precision:.4f}")
        print(f"Recall {label}: {recall:.4f}")
        print(f"PR_AUC {label}: {pr_auc:.4f}")
        print(f"F2-Score {label}: {f2_score:.4f}")
        
        if TAG:
            title = "_".join([TAG, label])
        else:
            title = label
        pmp.plot_roc(y_test, predictions[pred_label], title)
        pmp.plot_distributions(y_test, predictions[pred_label], title)
        threshold = pmp.find_threshold_max_f1(y_test, predictions[pred_label], title, N = 100)
        binary_predictions = np.where(predictions[pred_label] >= threshold, 1, 0)
        pmp.plot_confusion_matrix(y_test, binary_predictions, [0, 1], title)
        print("Done!", end="\n\n")

        if SAVE:
            print("Saving...")
            bst.save_model(os.path.join(MODELS_FOLDER, title + ".model"))
            print("Done!", end="\n\n")

    predictions = pd.DataFrame(predictions)
    predictions.to_csv(os.path.join(RESULTS_FOLDER, PREDICTIONS_FILE), sep=";", index=False)
