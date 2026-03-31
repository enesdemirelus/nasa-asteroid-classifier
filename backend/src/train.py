import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def import_dataframe():
    df = pd.read_csv(os.path.join(BASE_DIR, "../dataset/nasa.csv"))
    return df


def drop_unnecessary_columns(df):
    columns_to_drop = [
        "Neo Reference ID",
        "Name",
        "Est Dia in M(min)",
        "Est Dia in M(max)",
        "Est Dia in Miles(min)",
        "Est Dia in Miles(max)",
        "Est Dia in Feet(min)",
        "Est Dia in Feet(max)",
        "Close Approach Date",
        "Epoch Date Close Approach",
        "Relative Velocity km per hr",
        "Miss Dist.(Astronomical)",
        "Miss Dist.(lunar)",
        "Miss Dist.(miles)",
        "Orbit ID",
        "Orbit Determination Date",
        "Orbiting Body",
        "Equinox",
        "Epoch Osculation",
        "Miles per hour",
    ]

    df = df.drop(columns=columns_to_drop)

    return df


def preprocess_data(df):
    X = df.drop(columns=["Hazardous"])
    y = df["Hazardous"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler



def train_dataset(X_train, X_test, y_train, y_test, scaler):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    
    model.fit(X_train, y_train, epoch = 50, batch_size = 32, class_weight={0: 1, 1: 5},
    validation_split=0.2)
    
    model.evaluate(X_test, y_test)
    model.save(model.keras)
    joblib.dump(scaler, os.path.join(BASE_DIR, 'scaler.pkl'))
    joblib.dump(list(X.columns), os.path.join(BASE_DIR, 'feature_columns.pkl'))
    
    


if __name__ == "__main__":
    df = import_dataframe()
    df = drop_unnecessary_columns(df)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    train_dataset(X_train, X_test, y_train, y_test, scaler)
    print("Finished")
