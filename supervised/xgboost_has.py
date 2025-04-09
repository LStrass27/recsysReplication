import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import xgboost as xgb

from utils.pandas_utils import group_categoricals_tail
import utils.pimpmatplotlib as pm
from utils.xgbextras import stopping_at
from sklearn.metrics import roc_auc_score

# Config
CONFIG_FOLDER = os.path.join("config")
CONFIG_FILE = "features.csv"
DATA_FOLDER = os.path.join("..", "sample_data")
DATA_FILE = "dataset.csv"
MODELS_FOLDER = os.path.join("models")
RESULTS_FOLDER = os.path.join("results")

MODEL_TYPE = "xgboost"
TAG = "xgb"

PREDICTIONS_FILE = "_".join([MODEL_TYPE, TAG, "pred.csv"])

SAVE = True

# XGBoost params
# use an optimisation method to find the best params
SEED = 17
XGB_PARAMS = {
    "learning_rate": 0.1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "seed": SEED,
    "verbose": 0,
    "max_depth": 8,
    "min_child_weight": 3,
    "colsample_bytree": 0.75,
    "subsample": 0.75,
    "gamma": 5
}


if __name__ == "__main__":
    pmp = pm.PimpPlot(save=SAVE, folder=os.path.join(RESULTS_FOLDER, "plots"))

    # Load config
    source_config = os.path.join(CONFIG_FOLDER, CONFIG_FILE)
    features = pd.read_csv(source_config, keep_default_na=False, na_values=[""])
    print(features.columns)
    index = features.loc[features[MODEL_TYPE] == "index", "column"].tolist()
    print(index)
    predictors = features.loc[features[MODEL_TYPE] == "predictor", "column"].tolist()
    labels = features.loc[features[MODEL_TYPE] == "label", "column"].tolist()
    categorical = features.loc[(features["categorical"] == 1) & (features["column"].isin(predictors)), "column"].tolist()

    # Load data
    source_file = os.path.join(DATA_FOLDER, DATA_FILE)
    data = pd.read_csv(source_file, usecols=index+labels+predictors+["SET"],
                    sep=";", decimal=".", encoding="latin1",
                    keep_default_na = False, na_values = [""])

    # Preprocessing XGBoost
    group_categoricals_tail(data, categorical)
    data = pd.get_dummies(data, columns=categorical).copy()

    # Split the dataset
    indexes = {"train": None, "valid": None, "test": None}
    for set_name in indexes.keys():
        indexes[set_name] = np.where(data["SET"] == set_name)[0]

    # Get only relevant features
    xgb_features = [x for x in sorted(data.columns.tolist()) if x not in labels + index + ["SET"]]

    d = {}
    for set_name, set_indexes in indexes.items():
        d[set_name] = xgb.DMatrix(data.loc[set_indexes, xgb_features])

    predictions = {}
    for label in labels:
        print("----------------------------", end="\n")
        print(label, end="\n")
        print("----------------------------", end="\n\n")
        
        print("Creating the DMatrix...")
        for set_name, set_indexes in indexes.items():
            d[set_name].set_label(data.loc[set_indexes, label].values)
        print("Done!", end="\n\n")

        print("Training XGB...")
        bst = xgb.train(params=XGB_PARAMS, 
                        num_boost_round=3000, 
                        dtrain=d["train"], evals=[(d["valid"], "val")], 
                        callbacks=[stopping_at(5*10**(-4))])
        print("Done!", end="\n\n")

        print("Predictions and plots XGB...")
        pred_label = "{0}_PRED_{1}".format(label, TAG.upper())
        predictions[pred_label] = bst.predict(d["test"])
        
        print("Logloss: {}".format(log_loss(d["test"].get_label(), predictions[pred_label])), end="\n\n")
        auc_score = roc_auc_score(d["test"].get_label(), predictions[pred_label])
        print(f"AUC {label}: {auc_score:.4f}")
        
        if TAG:
            title = "_".join([TAG, label])
        else:
            title = label
        pmp.plot_roc(d["test"].get_label(), predictions[pred_label], title)
        pmp.plot_distributions(d["test"].get_label(), predictions[pred_label], title)
        threshold = pmp.find_threshold_max_f1(d["test"].get_label(), predictions[pred_label], title, N = 100)
        binary_predictions = np.where(predictions[pred_label] >= threshold, 1, 0)
        pmp.plot_confusion_matrix(d["test"].get_label(), binary_predictions, [0, 1], title)
        print("Done!", end="\n\n")

        if SAVE:
            print("Saving...")
            bst.save_model(os.path.join(MODELS_FOLDER, title + ".model"))
            print("Done!", end="\n\n")

    predictions = pd.DataFrame(predictions)
    predictions.to_csv(os.path.join(RESULTS_FOLDER, PREDICTIONS_FILE), sep=";", index=False)
