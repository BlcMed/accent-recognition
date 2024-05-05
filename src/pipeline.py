from sklearn.pipeline import Pipeline
from split_silence_transformer import SplitSilenceTransformer
from mfcc_transformer import MfccTransformer

pipeline = Pipeline(
    ("split_silence_transformer", SplitSilenceTransformer()),
    ("mfcc_transformer", MfccTransformer()) 
)