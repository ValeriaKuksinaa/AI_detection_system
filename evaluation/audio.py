import polars as pl
import librosa
import numpy as np 
from tqdm.auto import tqdm
import torch
import gc
import joblib 

def test_data(path_audio):
    dataset = pl.DataFrame({"path": path_audio})
    dataset = dataset.with_columns([
        (pl.Series([0.1]).alias("zero_crossing_rate")),
        (pl.Series([0.1]).alias("zero_crossings")),
        (pl.Series([0.1]).alias("spectrogram")),
        (pl.Series([0.1]).alias("mel_spectrogram")),
        (pl.Series([0.1]).alias("harmonics")),
        (pl.Series([0.1]).alias("perceptual_shock_wave")),
        (pl.Series([0.1]).alias("spectral_centroids")),
        (pl.Series([0.1]).alias("spectral_centroids_delta")),
        (pl.Series([0.1]).alias("spectral_centroids_accelerate")),
        (pl.Series([0.1]).alias("chroma1")),
        (pl.Series([0.1]).alias("chroma2")),
        (pl.Series([0.1]).alias("chroma3")),
        (pl.Series([0.1]).alias("chroma4")),
        (pl.Series([0.1]).alias("chroma5")),
        (pl.Series([0.1]).alias("chroma6")),
        (pl.Series([0.1]).alias("chroma7")),
        (pl.Series([0.1]).alias("chroma8")),
        (pl.Series([0.1]).alias("chroma9")),
        (pl.Series([0.1]).alias("chroma10")),
        (pl.Series([0.1]).alias("chroma11")),
        (pl.Series([0.1]).alias("chroma12")),
        (pl.Series([0.1]).alias("tempo_bpm")),
        (pl.Series([0.1]).alias("spectral_rolloff")),
        (pl.Series([0.1]).alias("spectral_flux")),
        (pl.Series([0.1]).alias("spectral_bandwidth_2")),
        (pl.Series([0.1]).alias("spectral_bandwidth_3")),
        (pl.Series([0.1]).alias("spectral_bandwidth_4")),
    ])
    for i in tqdm(range(len(dataset))):
        audio_path = dataset[i, 0]

        y, sr = librosa.load(audio_path)

        signal = librosa.effects.trim(y)[0]

        d_audio = np.abs(librosa.stft(signal, n_fft=512)) # рекомендованное значение https://librosa.org/doc/main/generated/librosa.stft.html
        db_audio = librosa.amplitude_to_db(d_audio, ref=np.max)

        s_audio = librosa.feature.melspectrogram(y = signal, sr=sr)
        s_db_audio = librosa.amplitude_to_db(s_audio, ref=np.max)

        y_harm, y_perc = librosa.effects.hpss(signal)

        spectral_centroids = librosa.feature.spectral_centroid(y = signal, sr=sr)[0]
        spectral_centroids_delta = librosa.feature.delta(spectral_centroids)
        spectral_centroids_accelerate = librosa.feature.delta(spectral_centroids, order=2)

        chromagram = librosa.feature.chroma_stft(y = signal, sr=sr)

        tempo_y = librosa.beat.beat_track(y = signal, sr=sr)[0]

        spectral_rolloff = librosa.feature.spectral_rolloff(y = signal, sr=sr)[0]

        onset_env = librosa.onset.onset_strength(y=signal, sr=sr)

        spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(y = signal, sr=sr)[0]
        spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(y = signal, sr=sr, p=3)[0]
        spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(y = signal, sr=sr, p=4)[0]

        dataset[i, "zero_crossing_rate"] = np.mean(librosa.feature.zero_crossing_rate(signal)[0])
        dataset[i, "zero_crossings"] = np.sum(librosa.zero_crossings(signal, pad=False))
        dataset[i, "spectrogram"] = np.mean(db_audio[0])
        dataset[i, "mel_spectrogram"] = np.mean(s_db_audio[0])
        dataset[i, "harmonics"] = np.mean(y_harm)
        dataset[i, "perceptual_shock_wave"] = np.mean(y_perc)
        dataset[i, "spectral_centroids"] = np.mean(spectral_centroids)
        dataset[i, "spectral_centroids_delta"] = np.mean(spectral_centroids_delta)
        dataset[i, "spectral_centroids_accelerate"] = np.mean(spectral_centroids_accelerate)
        dataset[i, "chroma1"] = np.mean(chromagram[0])
        dataset[i, "chroma2"] = np.mean(chromagram[1])
        dataset[i, "chroma3"] = np.mean(chromagram[2])
        dataset[i, "chroma4"] = np.mean(chromagram[3])
        dataset[i, "chroma5"] = np.mean(chromagram[4])
        dataset[i, "chroma6"] = np.mean(chromagram[5])
        dataset[i, "chroma7"] = np.mean(chromagram[6])
        dataset[i, "chroma8"] = np.mean(chromagram[7])
        dataset[i, "chroma9"] = np.mean(chromagram[8])
        dataset[i, "chroma10"]  = np.mean(chromagram[9])
        dataset[i, "chroma11"] = np.mean(chromagram[10])
        dataset[i, "chroma12"] = np.mean(chromagram[11])
        dataset[i, "tempo_bpm"] = tempo_y
        dataset[i, "spectral_rolloff"] = np.mean(spectral_rolloff)
        dataset[i, "spectral_flux"] = np.mean(onset_env)
        dataset[i, "spectral_bandwidth_2"] = np.mean(spectral_bandwidth_2)
        dataset[i, "spectral_bandwidth_3"] = np.mean(spectral_bandwidth_3)
        dataset[i, "spectral_bandwidth_4"] = np.mean(spectral_bandwidth_4)
    return dataset


def evaluation_audio(path_audio):
    data = test_data(path_audio)
    
    loaded_model = joblib.load("/home/rijkaa/leraa/solution/random_forest.joblib")
    X_val = data[:, 1:]
    prediction = loaded_model.predict(X_val)
    if prediction[0] > 0.5:
        print('Generated')
    else:
        print('Natural')
    
    torch.cuda.empty_cache()
    gc.collect()
