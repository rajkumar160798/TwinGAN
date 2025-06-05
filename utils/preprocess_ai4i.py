# utils/preprocess_ai4i.py
# Preprocess AI4I 2020 dataset into time-series windows

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def preprocess_ai4i(file_path, window_size=50, step=1, output_dir="data"):
    df = pd.read_csv(file_path)

    # Drop non-sensor columns
    sensor_cols = ['Air temperature [K]', 'Process temperature [K]',
                   'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

    # Normalize sensor data
    scaler = MinMaxScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])

    # Label 1 = failure
    labels = df['Machine failure'].values
    sensor_data = df[sensor_cols].values

    # Create sequences and corresponding labels
    sequences = []
    seq_labels = []

    for i in range(0, len(sensor_data) - window_size + 1, step):
        window = sensor_data[i:i+window_size]
        label_window = labels[i:i+window_size]
        sequences.append(window)
        seq_labels.append(1 if np.any(label_window == 1) else 0)

    sequences = np.array(sequences)
    seq_labels = np.array(seq_labels)

    # Save all sequences and failure-only sequences
    np.save(os.path.join(output_dir, "X_real.npy"), sequences)
    np.save(os.path.join(output_dir, "X_fail.npy"), sequences[seq_labels == 1])
    print(f"Saved {len(sequences)} total sequences and {sum(seq_labels)} failure sequences.")

if __name__ == "__main__":
    preprocess_ai4i("data/ai4i2020.csv")
