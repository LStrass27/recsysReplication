import os

from keras import backend as K
from keras.models import Sequential
from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Activation, Reshape, Concatenate, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.constraints import unit_norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf

from utils.pandas_utils import group_categoricals_tail
import utils.pimpmatplotlib as pm

# Keras config
KERAS_BACKEND = "GPU"
NUM_CORES = 4

if KERAS_BACKEND == "CPU":
    num_CPU = 1
    num_GPU = 0
else:
    num_GPU = 1
    num_CPU = 1
    
config = tf.ConfigProto(intra_op_parallelism_threads=NUM_CORES,
                        inter_op_parallelism_threads=NUM_CORES, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)

# Config
SEED = 17

CONFIG_FOLDER = os.path.join("config")
CONFIG_FILE = "features.csv"
DATA_FOLDER = os.path.join("..", "sample_data")
DATA_FILE = "dataset.csv"
MODELS_FOLDER = os.path.join("models")
RESULTS_FOLDER = os.path.join("results")

MODEL_TYPE = "ann"
TAG = "ee"

PREDICTIONS_FILE = "_".join([MODEL_TYPE, TAG, "pred.csv"])

SAVE = True


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
    binary = features.loc[(features["binary"] == 1) & (features["column"].isin(predictors)), "column"].tolist()
    numerical = features.loc[(features["numerical"] == 1) & (features["column"].isin(predictors)), "column"].tolist()
    categorical = features.loc[(features["categorical"] == 1) & (features["column"].isin(predictors)), "column"].tolist()

    # Load data
    source_file = os.path.join(DATA_FOLDER, DATA_FILE)
    data = pd.read_csv(source_file, usecols=index+labels+predictors+["SET"],
                    sep=";", decimal=".", encoding="latin1",
                    keep_default_na = False, na_values = [""])

    # Split the dataset
    indexes = {"train": None, "valid": None, "test": None}
    for set_name in indexes.keys():
        indexes[set_name] = np.where(data["SET"] == set_name)[0]

    # Create input datasets
    input_dict = {"train": {}, "valid": {}, "test": {}}

    # Preprocessing ann
    group_categoricals_tail(data, categorical)

    # Min Max scaling numerical variables
    columns = data.columns.tolist()
    for col in numerical:
        min_max_scaler = MinMaxScaler()
        data[col] = min_max_scaler.fit_transform(data[col].values.reshape(-1, 1))

    for set_name in input_dict.keys():
        input_dict[set_name]["numerical"] = data.loc[indexes[set_name], numerical].values

    # Label encoder categorical variables
    len_categorical = {}
    label_encoding = {}
    for col in categorical:
        unique_values = data[col].unique().tolist()
        len_categorical[col] = len(unique_values)
        label_encoding[col] = LabelEncoder()
        label_encoding[col].fit(sorted(unique_values))
        for set_name in input_dict.keys():
            input_dict[set_name][col] = label_encoding[col].transform(data.loc[indexes[set_name], col].values)

    # One-hot encoding over categoricals
    one_hot_data = pd.get_dummies(data[categorical], drop_first=True)

    for set_name in input_dict.keys():
        input_dict[set_name]["one_hot"] = one_hot_data.iloc[indexes[set_name], :]

    # Labels
    y = {
        "train": data.loc[indexes["train"], labels].values,
        "valid": data.loc[indexes["valid"], labels].values,
        "test": data.loc[indexes["test"], labels].values
    }

    # Keras model
    embedding_dim = {
        "DEMO_CAT_03": 3,
        "DEMO_CAT_04": 3,
        "DEMO_CAT_07": 3,
        "DEMO_CAT_09": 3,
        "DEMO_CAT_11": 3,
        "DEMO_CAT_12": 3,
        "DEMO_CAT_13": 3,
        "DEMO_CAT_14": 3,
        "DEMO_CAT_15": 3
    }

    # Embedding
    input_DEMO_CAT_03 = Input(shape=(1,), name="DEMO_CAT_03")
    output_DEMO_CAT_03 = Embedding(len_categorical["DEMO_CAT_03"], embedding_dim["DEMO_CAT_03"], name="DEMO_CAT_03_embedding", embeddings_constraint=unit_norm(axis=0))(input_DEMO_CAT_03)
    output_DEMO_CAT_03 = Reshape(target_shape=(embedding_dim["DEMO_CAT_03"],))(output_DEMO_CAT_03)

    input_DEMO_CAT_04 = Input(shape=(1,), name="DEMO_CAT_04")
    output_DEMO_CAT_04 = Embedding(len_categorical["DEMO_CAT_04"], embedding_dim["DEMO_CAT_04"], name="DEMO_CAT_04_embedding", embeddings_constraint=unit_norm(axis=0))(input_DEMO_CAT_04)
    output_DEMO_CAT_04 = Reshape(target_shape=(embedding_dim["DEMO_CAT_04"],))(output_DEMO_CAT_04)

    input_DEMO_CAT_07 = Input(shape=(1,), name="DEMO_CAT_07")
    output_DEMO_CAT_07 = Embedding(len_categorical["DEMO_CAT_07"], embedding_dim["DEMO_CAT_07"], name="DEMO_CAT_07_embedding", embeddings_constraint=unit_norm(axis=0))(input_DEMO_CAT_07)
    output_DEMO_CAT_07 = Reshape(target_shape=(embedding_dim["DEMO_CAT_07"],))(output_DEMO_CAT_07)

    input_DEMO_CAT_09 = Input(shape=(1,), name="DEMO_CAT_09")
    output_DEMO_CAT_09 = Embedding(len_categorical["DEMO_CAT_09"], embedding_dim["DEMO_CAT_09"], name="DEMO_CAT_09_embedding", embeddings_constraint=unit_norm(axis=0))(input_DEMO_CAT_09)
    output_DEMO_CAT_09 = Reshape(target_shape=(embedding_dim["DEMO_CAT_09"],))(output_DEMO_CAT_09)

    input_DEMO_CAT_11 = Input(shape=(1,), name="DEMO_CAT_11")
    output_DEMO_CAT_11 = Embedding(len_categorical["DEMO_CAT_11"], embedding_dim["DEMO_CAT_11"], name="DEMO_CAT_11_embedding", embeddings_constraint=unit_norm(axis=0))(input_DEMO_CAT_11)
    output_DEMO_CAT_11 = Reshape(target_shape=(embedding_dim["DEMO_CAT_11"],))(output_DEMO_CAT_11)

    input_DEMO_CAT_12 = Input(shape=(1,), name="DEMO_CAT_12")
    output_DEMO_CAT_12 = Embedding(len_categorical["DEMO_CAT_12"], embedding_dim["DEMO_CAT_12"], name="DEMO_CAT_12_embedding", embeddings_constraint=unit_norm(axis=0))(input_DEMO_CAT_12)
    output_DEMO_CAT_12 = Reshape(target_shape=(embedding_dim["DEMO_CAT_12"],))(output_DEMO_CAT_12)

    input_DEMO_CAT_13 = Input(shape=(1,), name="DEMO_CAT_13")
    output_DEMO_CAT_13 = Embedding(len_categorical["DEMO_CAT_13"], embedding_dim["DEMO_CAT_13"], name="DEMO_CAT_13_embedding", embeddings_constraint=unit_norm(axis=0))(input_DEMO_CAT_13)
    output_DEMO_CAT_13 = Reshape(target_shape=(embedding_dim["DEMO_CAT_13"],))(output_DEMO_CAT_13)

    input_DEMO_CAT_14 = Input(shape=(1,), name="DEMO_CAT_14")
    output_DEMO_CAT_14 = Embedding(len_categorical["DEMO_CAT_14"], embedding_dim["DEMO_CAT_14"], name="DEMO_CAT_14_embedding", embeddings_constraint=unit_norm(axis=0))(input_DEMO_CAT_14)
    output_DEMO_CAT_14 = Reshape(target_shape=(embedding_dim["DEMO_CAT_14"],))(output_DEMO_CAT_14)

    input_DEMO_CAT_15 = Input(shape=(1,), name="DEMO_CAT_15")
    output_DEMO_CAT_15 = Embedding(len_categorical["DEMO_CAT_15"], embedding_dim["DEMO_CAT_15"], name="DEMO_CAT_15_embedding", embeddings_constraint=unit_norm(axis=0))(input_DEMO_CAT_15)
    output_DEMO_CAT_15 = Reshape(target_shape=(embedding_dim["DEMO_CAT_15"],))(output_DEMO_CAT_15)

    input_numerical = Input(shape=(input_dict["train"]["numerical"].shape[1], ), name="numerical")
    input_one_hot = Input(shape=(input_dict["train"]["one_hot"].shape[1], ), name="one_hot")

    input_model = [
        input_DEMO_CAT_03,
        input_DEMO_CAT_04,
        input_DEMO_CAT_07,
        input_DEMO_CAT_09,
        input_DEMO_CAT_11,
        input_DEMO_CAT_12,
        input_DEMO_CAT_13,
        input_DEMO_CAT_14,
        input_DEMO_CAT_15,
        input_one_hot,
        input_numerical
    ]

    output_embeddings = [
        output_DEMO_CAT_03,
        output_DEMO_CAT_04,
        output_DEMO_CAT_07,
        output_DEMO_CAT_09,
        output_DEMO_CAT_11,
        output_DEMO_CAT_12,
        output_DEMO_CAT_13,
        output_DEMO_CAT_14,
        output_DEMO_CAT_15
    ]

    # Network graph
    output_model = Concatenate()(output_embeddings)
    output_model = Dense(512, kernel_initializer="uniform")(output_model)
    output_model = Activation("relu")(output_model)
    output_model = Concatenate()([output_model, input_numerical, input_one_hot])
    output_model = BatchNormalization()(output_model)
    output_model = Dense(256, kernel_initializer="uniform")(output_model)
    output_model = Activation("relu")(output_model)
    output_model = Dense(128, kernel_initializer="uniform")(output_model)
    output_model = Activation("relu")(output_model)
    output_model = Dense(32, kernel_initializer="uniform")(output_model)
    output_model = Activation("relu")(output_model)
    output_model = Dense(10)(output_model)
    output_model = Activation("sigmoid")(output_model)

    model = KerasModel(inputs=input_model, outputs=output_model)

    model.compile(loss="binary_crossentropy", optimizer="adam")

    callbacks_list = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=3
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=4
        )
    ]

    # Train
    history = model.fit(input_dict["train"], y["train"], 
                        validation_data=(input_dict["valid"], y["valid"]), 
                        batch_size=1024, epochs=50,
                        callbacks=callbacks_list)

    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Predict
    predictions = model.predict(input_dict["test"])
    predictions = pd.DataFrame(predictions)
    predictions.rename(columns={i:lab+"_PRED_EE" for i, lab in enumerate(labels)}, inplace=True)

    # Plots
    for index, label in enumerate(labels):
        if TAG:
            title = "_".join([TAG, label])
        else:
            title = label
        
        pmp.plot_roc(y["test"][:, index], predictions.values[:, index], title)
        pmp.plot_distributions(y["test"][:, index], predictions.values[:, index], title)
        threshold = pmp.find_threshold_max_f1(y["test"][:, index], predictions.values[:, index], title, N = 99)
        binary_predictions = np.where(predictions.values[:, index] >= threshold, 1, 0)
        pmp.plot_confusion_matrix(y["test"][:, index], binary_predictions, [0, 1], title)

    # Embedding results
    embedding_results = {"feature": [], "value": [], "x": [], "y": [], "z": []}
    for col in categorical:
        embedding = model.get_layer("{}_embedding".format(col)).get_weights()[0]
        text = label_encoding[col].inverse_transform([x for x in range(len_categorical[col])])
        for i in range(0, embedding.shape[0]):
            embedding_results["feature"].append(col)
            embedding_results["value"].append(text[i])
            embedding_results["x"].append(embedding[i, 0])
            embedding_results["y"].append(embedding[i, 1])
            embedding_results["z"].append(embedding[i, 2])
    embedding_df = pd.DataFrame(embedding_results)

    # Plot with plotly
    for col in categorical:
        embedding = model.get_layer("{}_embedding".format(col)).get_weights()[0]
        trace1 = go.Scatter3d(
            x=embedding[:,0],
            y=embedding[:,1],
            z=embedding[:,2],
            mode="markers+text",
            marker=dict(
                size=4,
                line=dict(
                    color="rgba(217, 217, 217, 0.14)",
                    width=0.5
                ),
                opacity=0.8
            ),
            text=label_encoding[col].inverse_transform([_x for _x in range(len_categorical[col])])
        )
        data_to_plot = [trace1]
        layout = go.Layout(
            title=col,
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            )
        )
        fig = go.Figure(data=data_to_plot, layout=layout)
        plotly.offline.plot(fig, filename=os.path.join(RESULTS_FOLDER, "plots", col + '.html'))

    # Save
    if SAVE:
        model.save(os.path.join(MODELS_FOLDER, TAG + ".model"))
        embedding_df.to_csv(os.path.join(RESULTS_FOLDER, TAG + ".csv"), sep=";", index=False)
        predictions.to_csv(os.path.join(RESULTS_FOLDER, PREDICTIONS_FILE), sep=";", index=False)
