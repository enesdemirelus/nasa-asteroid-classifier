import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os
import joblib
import matplotlib.pyplot as plt
 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_asteroid_dataset():
    df = pd.read_csv(os.path.join(BASE_DIR, "../dataset/nasa.csv"))
    return df




def remove_unused_columns(df):
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
    print(df.columns)

    return df


def split_and_scale_dataset(df):
    X = df.drop(columns=["Hazardous"])
    y = df["Hazardous"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler



def train_hazard_classifier(X_train, X_test, y_train, y_test):
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
    
    history = model.fit(X_train, y_train, epochs = 50, batch_size = 32, class_weight={0: 1, 1: 5},
    validation_split=0.2)
    
    model.evaluate(X_test, y_test)
    model.save("model.keras")
    
    return history


def save_preprocessing_artifacts(df, scaler):
    X = df.drop(columns=['Hazardous'])
    joblib.dump(scaler, os.path.join(BASE_DIR, 'scaler.pkl'))
    joblib.dump(list(X.columns), os.path.join(BASE_DIR, 'feature_columns.pkl'))
    
    
def plot_training_metrics(history):
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    
    precision = history.history["precision"]
    val_precision = history.history["val_precision"]
    
    recall = history.history["recall"]
    val_recall = history.history["val_recall"]
    
    epochs = range(1, len(accuracy) + 1)

    plt.figure()
    plt.plot(epochs, accuracy, label="accuracy")
    plt.plot(epochs, loss, label="loss")
    plt.plot(epochs, precision, label="precision")
    plt.plot(epochs, recall, label="recall")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.plot(epochs, val_precision, label="val_precision")
    plt.plot(epochs, val_recall, label="val_recall")
    plt.legend()
    plt.show()
    
    
    
if __name__ == "__main__":
    df = load_asteroid_dataset()
    df = remove_unused_columns(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale_dataset(df)
    history = train_hazard_classifier(X_train, X_test, y_train, y_test)
    save_preprocessing_artifacts(df, scaler)
    plot_training_metrics(history)
    print("Finished")
