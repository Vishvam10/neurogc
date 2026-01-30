import os
import json
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def load_config(config_path: str = "config.json") -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)


class GCDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        sequence_length: int = 10,
        feature_columns: Optional[list[str]] = None,
        normalize: bool = True
    ):

        self.sequence_length = sequence_length
        self.normalize = normalize
        
        self.feature_columns = feature_columns or [
            'cpu', 'mem', 'disk_read', 'disk_write',
            'net_sent', 'net_recv', 'rps', 'p95', 'p99', 'gc_triggered'
        ]
        
        self.df = pd.read_csv(csv_path)
        
        if 'gc_triggered' in self.df.columns:
            self.df['gc_triggered'] = self.df['gc_triggered'].astype(int)
        
        self.features = self.df[self.feature_columns].values.astype(np.float32)
        
        # Normalization parameters
        self.feature_means = None
        self.feature_stds = None
        
        if self.normalize:
            self.feature_means = self.features.mean(axis=0)
            self.feature_stds = self.features.std(axis=0)
            # Avoid division by zero
            self.feature_stds[self.feature_stds == 0] = 1.0
            self.features = (self.features - self.feature_means) / self.feature_stds
        
        self._create_targets()

    # Peak heuristics ik ..
    def _create_targets(self) -> None:

        df = self.df
        
        # Combine memory pressure, allocation rate, and existing GC activity
        mem_pressure = df['mem'].values / 100.0 
        
        # Normalize other metrics to [0, 1] range
        cpu_factor = df['cpu'].values / 100.0
        
        # Higher memory + higher CPU = more likely need GC
        # If GC was recently triggered, urgency is lower
        gc_recent = df['gc_triggered'].astype(float).values
        
        # Simple heuristic formula for target
        # High memory + high CPU + no recent GC = high urgency
        self.targets = np.clip(
            0.4 * mem_pressure + 0.3 * cpu_factor + 0.3 * (1 - gc_recent * 0.5),
            0.0, 1.0
        ).astype(np.float32)

    def __len__(self) -> int:
        return max(0, len(self.features) - self.sequence_length)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Input: sequence of features
        x = torch.tensor(self.features[idx:idx + self.sequence_length])
        
        # Target: GC urgency at the end of the sequence
        y = torch.tensor([self.targets[idx + self.sequence_length - 1]])
        
        return x, y

    def get_normalization_params(self) -> tuple[np.ndarray, np.ndarray]:
        """Return normalization parameters (means, stds)."""
        return self.feature_means, self.feature_stds


class GCPredictor(nn.Module):
    """
    LSTM-based model for predicting GC urgency.
    
    Architecture:
        Input -> LSTM -> Linear -> Sigmoid -> Output [0, 1]
    """

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        out = self.sigmoid(out)
        
        return out


class GCPredictorWrapper:
   
    def __init__(
        self,
        model: GCPredictor,
        feature_means: Optional[np.ndarray] = None,
        feature_stds: Optional[np.ndarray] = None,
        sequence_length: int = 10,
        device: str = "cpu"
    ):

        self.model = model
        self.model.eval()
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        self.sequence_length = sequence_length
        self.device = device
        self.model.to(device)
        
        # Buffer for building sequences
        self._buffer: list[np.ndarray] = []

    def add_metrics(self, metrics: dict) -> None:
        feature_order = [
            'cpu', 'mem', 'disk_read', 'disk_write',
            'net_sent', 'net_recv', 'rps', 'p95', 'p99', 'gc_triggered'
        ]
        
        features = np.array([
            float(metrics.get(k, 0)) for k in feature_order
        ], dtype=np.float32)
        
        self._buffer.append(features)
        
        # Keep only the last sequence_length entries
        if len(self._buffer) > self.sequence_length:
            self._buffer = self._buffer[-self.sequence_length:]

    def can_predict(self) -> bool:
        return len(self._buffer) >= self.sequence_length

    def predict(self) -> float:
        if not self.can_predict():
            return 0.0
        
        features = np.array(self._buffer[-self.sequence_length:])
        
        if self.feature_means is not None and self.feature_stds is not None:
            features = (features - self.feature_means) / self.feature_stds
        
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(x)
        
        return output.item()

    def predict_from_sequence(self, sequence: list[dict]) -> float:
        if len(sequence) < self.sequence_length:
            return 0.0
        
        self._buffer.clear()
        for metrics in sequence[-self.sequence_length:]:
            self.add_metrics(metrics)
        
        return self.predict()


def train_model(
    csv_path: str,
    config_path: str = "config.json",
    model_save_path: Optional[str] = None
) -> tuple[GCPredictor, float, dict]:

    config = load_config(config_path)
    lstm_params = config.get('lstm_params', {})
    
    # Extract parameters
    input_size = lstm_params.get('input_size', 10)
    hidden_size = lstm_params.get('hidden_size', 64)
    num_layers = lstm_params.get('num_layers', 2)
    sequence_length = lstm_params.get('sequence_length', 10)
    epochs = lstm_params.get('epochs', 100)
    learning_rate = lstm_params.get('learning_rate', 0.001)
    batch_size = lstm_params.get('batch_size', 32)
    
    # Create dataset
    dataset = GCDataset(csv_path, sequence_length=sequence_length)
    
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty or too small. Need at least {sequence_length + 1} rows.")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GCPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    final_loss = 0.0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / max(num_batches, 1)
        final_loss = avg_loss
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")
    
    # Get normalization parameters
    feature_means, feature_stds = dataset.get_normalization_params()
    norm_params = {
        'feature_means': feature_means.tolist() if feature_means is not None else None,
        'feature_stds': feature_stds.tolist() if feature_stds is not None else None,
        'sequence_length': sequence_length
    }
    
    # Save model
    save_path = model_save_path or config.get('model_path', 'gc_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers
        },
        'norm_params': norm_params
    }, save_path)
    print(f"Model saved to {save_path}")
    
    return model, final_loss, norm_params


def load_model(
    model_path: str,
    device: str = "cpu"
) -> GCPredictorWrapper:

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract config
    model_config = checkpoint['config']
    norm_params = checkpoint.get('norm_params', {})
    
    # Create model
    model = GCPredictor(
        input_size=model_config['input_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get normalization params
    feature_means = norm_params.get('feature_means')
    feature_stds = norm_params.get('feature_stds')
    sequence_length = norm_params.get('sequence_length', 10)
    
    if feature_means:
        feature_means = np.array(feature_means, dtype=np.float32)
    if feature_stds:
        feature_stds = np.array(feature_stds, dtype=np.float32)
    
    return GCPredictorWrapper(
        model=model,
        feature_means=feature_means,
        feature_stds=feature_stds,
        sequence_length=sequence_length,
        device=device
    )

# NOTE : For proper sequence-based prediction, use GCPredictorWrapper. 
# This method is for quick prediction function using a single metrics snapshot.
def predict(metrics: dict, model_path: str = "gc_model.pth") -> float:
    wrapper = load_model(model_path)
    
    # Fill buffer with repeated current metrics (not ideal, but works for 
    # single snapshot)
    for _ in range(wrapper.sequence_length):
        wrapper.add_metrics(metrics)
    
    return wrapper.predict()


if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser(description="Train or test GC predictor model")
    parser.add_argument("--train", type=str, help="Path to CSV file for training")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--model", type=str, default="gc_model.pth", help="Path to model file")
    parser.add_argument("--test", action="store_true", help="Test the model with sample data")
    
    args = parser.parse_args()
    
    if args.train:
        print(f"Training model from {args.train}...")
        model, loss, params = train_model(args.train, args.config, args.model)
        print(f"Training complete. Final loss: {loss:.6f}")
    
    elif args.test:
        print("Testing model with sample metrics...")
        sample_metrics = {
            'cpu': 45.0,
            'mem': 60.0,
            'disk_read': 1000000.0,
            'disk_write': 500000.0,
            'net_sent': 100000.0,
            'net_recv': 200000.0,
            'rps': 100.0,
            'p95': 50.0,
            'p99': 100.0,
            'gc_triggered': False
        }
        
        try:
            prediction = predict(sample_metrics, args.model)
            print(f"GC Urgency Prediction: {prediction:.4f}")
        except FileNotFoundError:
            print(f"Model file {args.model} not found. Train a model first.")
    
    else:
        parser.print_help()
