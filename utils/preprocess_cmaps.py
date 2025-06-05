# utils/preprocess_cmaps.py
# Basic CMAPSS preprocessing to extract windows

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_cmaps(path, window_size=50):
    df = pd.read_csv(path)
    useful_features = [col for col in df.columns if "sensor" in col]
    scaler = MinMaxScaler()
    df[useful_features] = scaler.fit_transform(df[useful_features])

    sequences = []
    for unit in df['unit'].unique():
        unit_df = df[df['unit'] == unit]
        for i in range(len(unit_df) - window_size):
            seq = unit_df[useful_features].iloc[i:i+window_size].values
            sequences.append(seq)

    return np.array(sequences)
