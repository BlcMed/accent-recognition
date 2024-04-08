# Accent detection DL model

## Data source

Data is collected from Accent Speech Archive

## Dependencies

The primary library used is Librosa

## Conventions to easily follow code

- when loading a file with librosa.load, the samples array is usually refered to as 'y', however i refere to it as 'waveform'
since this is about a model for accent recognition, the label is the accent itself, the audio is simply audio = (waveform, label)
