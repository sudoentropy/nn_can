"""
running cancer data through a neural network
based on the structured data classification tutorial on keras.io
"""


import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding


"""
# setting up data
"""


# load df
df = pd.read_csv("", sep=",")


# split train and test data
val_df = df.sample(frac=0.1, random_state=133780085)
train_df = df.drop(val_df.index)

print(
    "using %d samples for training and %d for validation"
    % (len(train_df), len(val_df))
)


# generating dataset objects for each dataframe
def dataframe_to_dataset(df):
    df = df.copy()
    labels = df.pop("duration")
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.shuffle(buffer_size=len(df))
    return ds


train_ds = dataframe_to_dataset(train_df)
val_ds = dataframe_to_dataset(val_df)


# input in dict of features, target is duration
for x, y in train_ds.take(1):
    print("Input: ", x)
    print("Duration: ", y)


# batching the datasets
train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)


"""
# preprocessing data
"""


# numerical data
def encode_numerical_feature(feature, name, dataset):
    normalizer = Normalization()

    # prepare dataset that only yields desired features
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # learn data statistics
    normalizer.adapt(feature_ds)

    # normalize input features
    encoded_feature = normalizer(feature)
    return encoded_feature


# binary integer data
def encode_integer_categorical_feature(feature, name, dataset):

    # encoding binary indicies
    encoder = CategoryEncoding(output_mode="binary")

    # prepare dataset that only yields desired features
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # learn space of inicies
    encoder.adapt(feature_ds)

    # one hot encoder
    encoded_feature = encoder(feature)
    return encoded_feature


"""
# building end to end model
"""


# categorical features coded as binary integers
gender = keras.Input(shape=(1,), name="gender", dtype="int64")
tp53 = keras.Input(shape=(1,), name="tp53", dtype="int64")
SMAD4 = keras.Input(shape=(1,), name="SMAD4", dtype="int64")
CDKN2A = keras.Input(shape=(1,), name="CDKN2A", dtype="int64")
KRAS = keras.Input(shape=(1,), name="KRAS", dtype="int64")
G12V = keras.Input(shape=(1,), name="G12V", dtype="int64")
G12D = keras.Input(shape=(1,), name="G12D", dtype="int64")
G12R = keras.Input(shape=(1,), name="G12R", dtype="int64")


# numerical features
age = keras.Input(shape=(1,), name="age")

all_inputs = [
    gender,
    age,
    tp53,
    SMAD4,
    CDKN2A,
    KRAS,
    G12V,
    G12R,
    G12D,
]


# binary integer categorical features
gender_encoded = encode_numerical_feature(gender, "gender", train_ds)
tp53_encoded = encode_numerical_feature(tp53, "tp53", train_ds)
SMAD4_encoded = encode_numerical_feature(SMAD4, "SMAD4", train_ds)
CDKN2A_encoded = encode_numerical_feature(CDKN2A, "CDKN2A", train_ds)
KRAS_encoded = encode_numerical_feature(KRAS, "KRAS", train_ds)
G12V_encoded = encode_numerical_feature(G12V, "G12V", train_ds)
G12D_encoded = encode_numerical_feature(G12D, "G12D", train_ds)
G12R_encoded = encode_numerical_feature(G12R, "G12R", train_ds)


# numerical features
age_encoded = encode_numerical_feature(age, "age", train_ds)


all_features = layers.concatenate(
    [
        gender_encoded,
        tp53_encoded,
        SMAD4_encoded,
        CDKN2A_encoded,
        KRAS_encoded,
        G12V_encoded,
        G12D_encoded,
        G12R_encoded,
        age_encoded,
    ]
)

x = layers.Dense(32, activation="relu")(all_features)
# x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(all_inputs, output)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])


# visualizing connectivity - this is throwing an error, so commented out for now
# keras.utils.plot_model(model, show_shapes=True, rankdir="LR")


"""
#  train the model
"""


model.fit(train_ds, epochs=5, validation_data=val_ds)
