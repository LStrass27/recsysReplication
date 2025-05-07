import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    average_precision_score,
    fbeta_score,
    jaccard_score
)

import utils.pimpmatplotlib as pm
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

# Config
CONFIG_FOLDER = os.path.join("config")
CONFIG_FILE = "features.csv"
DATA_FOLDER = os.path.join("sample_data")
DATA_FILE = "dataset.csv"
MODELS_FOLDER = os.path.join("models")
RESULTS_FOLDER = os.path.join("results")

MODEL_TYPE = "model"
TAG = "als"

PREDICTIONS_FILE = "_".join([MODEL_TYPE, TAG, "pred.csv"])

SAVE = True

if __name__ == "__main__":

    pmp = pm.PimpPlot(save=SAVE, folder=os.path.join(RESULTS_FOLDER, "plots"))

    # Load config
    source_config = os.path.join(CONFIG_FOLDER, CONFIG_FILE)
    features = pd.read_csv(source_config, keep_default_na=False, na_values=[""])
    index = features.loc[features[MODEL_TYPE] == "index", "column"].tolist()
    labels = features.loc[features[MODEL_TYPE] == "label", "column"].tolist()

    # Load data
    source_file = os.path.join(DATA_FOLDER, DATA_FILE)
    data = pd.read_csv(source_file, usecols=index+labels+["SET"],
                    sep=";", decimal=".", encoding="latin1",
                    keep_default_na = False, na_values = [""])

    # Split the dataset
    indexes = {"train": None, "valid": None, "test": None}
    for set_name in indexes.keys():
        indexes[set_name] = np.where(data["SET"] == set_name)[0]

    print(data.columns)

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
        
        train_idx = indexes["train"]
        train_values = data.loc[train_idx, label].astype(float).values
        train_user_ids = data.loc[train_idx, "UNIQUE_ID"].values

        interactions = csr_matrix(
            (train_values, (train_user_ids, np.zeros_like(train_user_ids))),
            shape=(data["UNIQUE_ID"].max() + 1, 1)
        )
        
        #print(interactions)
        
        print("Fitting ALS...")
        als = AlternatingLeastSquares(factors=20, regularization=0.1, iterations=15)
        als.fit(interactions.T)  # Transpose: implicit expects items × users

        print("Scoring test users...")
        test_user_ids = data.loc[indexes["test"], "UNIQUE_ID"].values

        user_factors = als.item_factors
        item_vector = als.user_factors[0]
        test_user_ids = data.loc[indexes["test"], "UNIQUE_ID"].values
        pred_scores = user_factors[test_user_ids] @ item_vector

        true_labels = data.loc[indexes["test"], label].values

        threshold_preds = (pred_scores > 0.5).astype(int)

        print(true_labels)
        print(threshold_preds)

        auc_score = roc_auc_score(true_labels, pred_scores)
        accuracy = accuracy_score(true_labels, threshold_preds)
        precision = precision_score(true_labels, threshold_preds)
        recall = recall_score(true_labels, threshold_preds)
        pr_auc = average_precision_score(true_labels, pred_scores)
        f2_score = fbeta_score(true_labels, threshold_preds, beta=2)
        
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
        pmp.plot_roc(true_labels, pred_scores, title)
        pmp.plot_distributions(true_labels, pred_scores, title)
        threshold = pmp.find_threshold_max_f1(true_labels, pred_scores, title, N = 100)
        binary_predictions = np.where(pred_scores >= threshold, 1, 0)
        pmp.plot_confusion_matrix(true_labels, binary_predictions, [0, 1], title)
        print("Done!", end="\n\n")

        #if SAVE:
        #    print("Saving...")
        #    model.save_model(os.path.join(MODELS_FOLDER, title + ".model"))
        #    print("Done!", end="\n\n")

    predictions = pd.DataFrame(predictions)
    predictions.to_csv(os.path.join(RESULTS_FOLDER, PREDICTIONS_FILE), sep=";", index=False)