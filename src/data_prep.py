from scipy.io.arff import loadarff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch

def load_ecg_data(train_file, test_file):
    train_data, train_meta = loadarff(open(train_file, 'r'))
    test_data, test_meta = loadarff(open(test_file, 'r'))
    train_df = pd.DataFrame(train_data)
    train_df['target'] = train_df.iloc[:, -1]
    test_df = pd.DataFrame(test_data)
    test_df['target'] = test_df.iloc[:, -1]
    df = pd.concat([train_df, test_df])
    normal_df = df[df['target'] == b'1'].drop('target', axis=1)
    anomaly_df = df[df['target'] != b'1'].drop('target', axis=1)
    return normal_df, anomaly_df

def create_dataset(normal_df, anomaly_df):
    # Each row is ALREADY a 140-timestep sequence - NO windowing needed!
    normal_seqs = normal_df.values.astype(np.float32)  # [N, 140]
    anomaly_seqs = anomaly_df.values.astype(np.float32)  # [M, 140]
    
    normal_train, normal_temp = train_test_split(normal_seqs, test_size=0.2, shuffle=False)
    normal_val, normal_test = train_test_split(normal_temp, test_size=0.5, shuffle=False)
    
    # Add channel dim: [N,140] → [N,1,140]
    train_dataset = torch.from_numpy(normal_train).unsqueeze(1)
    val_dataset = torch.from_numpy(normal_val).unsqueeze(1)
    test_normal = torch.from_numpy(normal_test).unsqueeze(1)
    test_anomaly = torch.from_numpy(anomaly_seqs).unsqueeze(1)
    
    return train_dataset, val_dataset, test_normal, test_anomaly
