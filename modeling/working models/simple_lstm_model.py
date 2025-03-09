import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

# Define a simple LSTM model using PyTorch Lightning
class SimpleLSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size=16, num_layers=1, lr=0.03):
        super().__init__()
        self.lr = lr
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Forecasting a single value
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use the output from the last time step
        return self.fc(lstm_out[:, -1, :])
    
    def training_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            batch = batch[0]
        x = batch["encoder_cont"]
        # Use the decoder target (which holds the future target "Promotions")
        y = batch.get("decoder_target")
        if y is None:
            raise KeyError(f"'decoder_target' not found in batch; available keys: {list(batch.keys())}")
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def main():
    # Read and preprocess data
    data = pd.read_csv(r"C:\Users\domdd\Documents\GitHub\SportsAnalysis\data\price_sales_promotion.csv")
    data["YearMonth"] = pd.to_datetime(data["YearMonth"], format="%Y%m")
    data["time_idx"] = data["YearMonth"].dt.year * 12 + data["YearMonth"].dt.month
    data["time_idx"] -= data["time_idx"].min()
    data["month"] = data["YearMonth"].dt.month.astype(str).astype("category")
    data["Sales"] = data["Sales"].clip(lower=1e-8)
    data["log_sales"] = np.log(data["Sales"])
    data["avg_sales_by_sku"] = data.groupby(["time_idx", "SKU"], observed=True)["Sales"].transform("mean")
    data["avg_sales_by_agency"] = data.groupby(["time_idx", "Agency"], observed=True)["Sales"].transform("mean")
    
    # Define time series parameters
    max_encoder_length = 12
    max_prediction_length = 1

    # Create the dataset
    training = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="Promotions",
        group_ids=["Agency", "SKU"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx", "Price", "Sales", "log_sales", "avg_sales_by_sku", "avg_sales_by_agency"],
        time_varying_known_categoricals=["month"],
        time_varying_unknown_reals=["Promotions"],
        target_normalizer=GroupNormalizer(groups=["Agency", "SKU"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # Create dataloader using the default collate function
    batch_size = 32
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=16
    )
    
    # Determine input size from a sample batch
    sample_batch = next(iter(train_dataloader))
    if isinstance(sample_batch, tuple):
        sample_batch = sample_batch[0]
    input_size = sample_batch["encoder_cont"].shape[-1]
    
    # Instantiate the LSTM model
    lstm_model = SimpleLSTM(input_size=input_size, hidden_size=16, num_layers=1, lr=0.03)
    
    # Set up trainer and train
    trainer = pl.Trainer(max_epochs=1, accelerator="gpu", devices=1)
    trainer.fit(lstm_model, train_dataloader)
    
    # For prediction, create a dataloader from the dataset with predict=True
    future_data = TimeSeriesDataSet.from_dataset(training, data, predict=True)
    future_dataloader = future_data.to_dataloader(train=False, batch_size=batch_size, num_workers=16)
    
    predictions = []
    lstm_model.eval()
    with torch.no_grad():
        for batch in future_dataloader:
            if isinstance(batch, tuple):
                batch = batch[0]
            x = batch["encoder_cont"]
            preds = lstm_model(x)
            predictions.append(preds)
    predictions = torch.cat(predictions, dim=0)
    
    print("Simple LSTM Model Predictions:")
    print(predictions)

if __name__ == "__main__":
    main()
