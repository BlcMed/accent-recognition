from data_manager import load_audio_files, filter_data_based_on_accents
from preprocess_pipeline import pipeline
from load_config import load_constants_from_yaml

from sklearn.model_selection import train_test_split
import pandas as pd

constants = load_constants_from_yaml('constants.yml')

sampling_rating = constants["SAMPLING_RATING"]
considered_accents = constants["CONSIDERED_ACCENTS"]

constants = load_constants_from_yaml('constants.yml')
batch_size = constants["BATCH_SIZE"]
epochs = constants["EPOCHS"]
validation_split = constants["VALIDATION_SPLIT"]
test_size = constants["TEST_SIZE"]
random_state = constants["RANDOM_STATE"]
processed_data_path = constants["PROCESSED_DATA_PATH"]


def run_preprocessing():
    df = load_audio_files("data/raw/recordings/", sr = sampling_rating)
    df = filter_data_based_on_accents(df=df, considered_accents=considered_accents)
    df_transformed = pipeline.fit_transform(df)
    return df_transformed

def train_model():
    df = pd.read_csv(processed_data_path)
    X = df.drop("label", axis=1)
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = train_model(X_train.values, y_train.values, batch_size, epochs, validation_split)
    return model

def run_full_pipeline():
    df_transformed = run_preprocessing()
    df_transformed.to_csv(processed_data_path, index = False)
    model = train_model()
    model.save("models/model.pth")


if __name__ == "__main__":
    run_preprocessing()