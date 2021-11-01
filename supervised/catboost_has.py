import os

from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

MODEL_TYPE = "catboost"
TAG = "ctb"

PREDICTIONS_FILE = "_".join([MODEL_TYPE, TAG, "pred.csv"])

SAVE = True

# catboost params
# use an optimization method to find the best params
SEED = 17
CTB_PARAMS = {
    "random_seed": SEED,
    "num_boost_round": 3000, 
    "depth": 8, 
    "learning_rate": 0.1, 
    "loss_function": "Logloss",
    "logging_level": "Verbose",
    "early_stopping_rounds": 5,
}


if __name__ == "__main__":

    pmp = pm.PimpPlot(save=SAVE, folder=os.path.join(RESULTS_FOLDER, "plots"))

    # Load config
    source_config = os.path.join(CONFIG_FOLDER, CONFIG_FILE)
    features = pd.read_csv(source_config,
                        sep=";", decimal=".", encoding="latin1",
                        keep_default_na = False, na_values = [""])
    index = features.loc[features[MODEL_TYPE] == "index", "column"].tolist()
    predictors = features.loc[features[MODEL_TYPE] == "predictor", "column"].tolist()
    labels = features.loc[features[MODEL_TYPE] == "label", "column"].tolist()
    categorical = features.loc[(features["categorical"] == 1) & (features["column"].isin(predictors)), "column"].tolist()

    # Load data
    source_file = os.path.join(DATA_FOLDER, DATA_FILE)
    data = pd.read_csv(source_file, usecols=index+labels+predictors+["SET"],
                    sep=";", decimal=".", encoding="latin1",
                    keep_default_na = False, na_values = [""])

    # Preprocessing catboost
    group_categoricals_tail(data, categorical)

    # Split the dataset
    indexes = {"train": None, "valid": None, "test": None}
    for set_name in indexes.keys():
        indexes[set_name] = np.where(data["SET"] == set_name)[0]

    # Get only relevant features
    ctb_features = [x for x in sorted(data.columns.tolist()) if x not in labels + index + ["SET"]]
    cat_features = [i for i, f in enumerate(ctb_features) if f in categorical]

    # catboost
    d = {}
    for set_name, set_indexes in indexes.items():
        d[set_name] = data.loc[set_indexes, ctb_features].values

    predictions = {}
    for label in labels:
        print("----------------------------", end="\n")
        print(label, end="\n")
        print("----------------------------", end="\n\n")
        
        print("Creating the Dataset...")
        y = {}
        for set_name, set_indexes in indexes.items():
            y[set_name] = data.loc[set_indexes, label].values

        print("Done!", end="\n\n")
        
        print("Training catboost...")
        model = CatBoostClassifier(**CTB_PARAMS)

        model.fit(d["train"], y["train"], 
                    cat_features=cat_features, 
                    eval_set=[(d["valid"], y["valid"])])
        print("Done!", end="\n\n")

        print("Predictions and plots CTB...")
        pred_label = "{0}_PRED_{1}".format(label, TAG.upper())
        predictions[pred_label] = model.predict_proba(d["test"])[:, 1]
        
        print("Logloss: {}".format(log_loss(y["test"], predictions[pred_label])), end="\n\n")
        
        if TAG:
            title = "_".join([TAG, label])
        else:
            title = label
        pmp.plot_roc(y["test"], predictions[pred_label], title)
        pmp.plot_distributions(y["test"], predictions[pred_label], title)
        threshold = pmp.find_threshold_max_f1(y["test"], predictions[pred_label], title, N = 100)
        binary_predictions = np.where(predictions[pred_label] >= threshold, 1, 0)
        pmp.plot_confusion_matrix(y["test"], binary_predictions, [0, 1], title)
        print("Done!", end="\n\n")

        if SAVE:
            print("Saving...")
            model.save_model(os.path.join(MODELS_FOLDER, title + ".model"))
            print("Done!", end="\n\n")

    predictions = pd.DataFrame(predictions)
    predictions.to_csv(os.path.join(RESULTS_FOLDER, PREDICTIONS_FILE), sep=";", index=False)
