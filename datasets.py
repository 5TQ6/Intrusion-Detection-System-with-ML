import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class WEBIDS23(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the WEB-IDS23 csv files.
        """
        self.root_dir = root_dir
        self.data_files = [f for f in os.listdir(root_dir) if f.endswith('.csv')]
        
        if not self.data_files:
            raise RuntimeError("No CSV files found in the specified directory.")

        self.features, self.labels = self._load_data()

    def _load_data(self):
        all_data = []
        for file_name in self.data_files:
            file_path = os.path.join(self.root_dir, file_name)
            df = pd.read_csv(file_path)
            all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Separate features and labels
        labels = combined_df['attack_type']
        features = combined_df.drop(columns=['uid', 'ts', 'id.orig_h', 'id.resp_h', 'service', 'traffic_direction', 'attack', 'attack_type'])

        # Convert categorical labels to numeric
        self.label_map = {label: i for i, label in enumerate(labels.unique())}
        numeric_labels = labels.map(self.label_map)

        # Convert feature columns to numeric, coercing errors
        numeric_features = features.apply(pd.to_numeric, errors='coerce')
        
        # Fill NaN values with 0. A better strategy might be imputation (e.g., mean, median).
        numeric_features = numeric_features.fillna(0)

        # Convert to PyTorch tensors
        features_tensor = torch.tensor(numeric_features.values, dtype=torch.float32)
        labels_tensor = torch.tensor(numeric_labels.values, dtype=torch.long)
        
        return features_tensor, labels_tensor

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

