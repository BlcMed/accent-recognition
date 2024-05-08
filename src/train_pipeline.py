from data_manager import load_audio_files, filter_data_based_on_accents
from pipeline import make_pipeline
from load_config import load_constants_from_yaml

constants = load_constants_from_yaml('constants.yml')

SAMPLING_RATING = constants["SAMPLING_RATING"]
FRAME_LENGTH_ENERGY = constants["FRAME_LENGTH_ENERGY"]
THRESHOLD_PERCENTAGE = constants["THRESHOLD_PERCENTAGE"]
MIN_SILENCE_DURATION = constants["MIN_SILENCE_DURATION"]
HOP_LENGTH = constants["HOP_LENGTH"]
SEGMENT_DURATION = constants["SEGMENT_DURATION"]
SEGMENT_OVERLAP = constants["SEGMENT_OVERLAP"]
N_MFCC = constants["N_MFCC"]
CONSIDERED_ACCENTS = constants["CONSIDERED_ACCENTS"]


def run_training():
    df = load_audio_files("data/raw/recordings/", sr = SAMPLING_RATING)
    df = filter_data_based_on_accents(df=df, considered_accents=CONSIDERED_ACCENTS)
    pipeline=make_pipeline(sampling_rating=SAMPLING_RATING,
                           min_silence_duration=MIN_SILENCE_DURATION,
                           threshold_percentage=THRESHOLD_PERCENTAGE,
                           hop_length=HOP_LENGTH,
                           frame_length_energy=FRAME_LENGTH_ENERGY,
                           segment_duration=SEGMENT_DURATION,
                           
                           )
    df_transformed = pipeline.fit_transform(df)
    df_transformed.to_csv('df_transformed.csv')


if __name__ == "__main__":
    run_training()