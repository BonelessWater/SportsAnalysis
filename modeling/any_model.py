import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.models.baseline import Baseline
from pytorch_forecasting.models.deepar import DeepAR
from pytorch_forecasting.models.mlp import DecoderMLP
from pytorch_forecasting.models.nbeats import NBeats
from pytorch_forecasting.models.nhits import NHiTS
from pytorch_forecasting.models.rnn import RecurrentNetwork
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss, MAE, NormalDistributionLoss

# Define a simple LSTM model using PyTorch Lightning
class SimpleLSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size=16, num_layers=1, lr=0.03):
        super().__init__()
        self.lr = lr
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Forecast a single value

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use the output from the last time step
        return self.fc(lstm_out[:, -1, :])
    
    def _get_input(self, batch):
        # Use "encoder_cont" if available; otherwise, use "encoder_target"
        if "encoder_cont" in batch:
            return batch["encoder_cont"]
        elif "encoder_target" in batch:
            return batch["encoder_target"]
        else:
            raise KeyError("Neither 'encoder_cont' nor 'encoder_target' found in batch.")
    
    def training_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            batch = batch[0]
        x = self._get_input(batch)
        y = batch.get("decoder_target")
        if y is None:
            raise KeyError(f"'decoder_target' not found in batch; available keys: {list(batch.keys())}")
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            batch = batch[0]
        x = self._get_input(batch)
        y = batch.get("decoder_target")
        if y is None:
            raise KeyError(f"'decoder_target' not found in batch; available keys: {list(batch.keys())}")
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def run_forecasting_model(model_name: str, data_path: str, max_epochs: int = 5, batch_size: int = 2):
    """
    Runs the specified forecasting model on the given dataset.

    Args:
        model_name (str): Name of the model to use. Options are:
            "Baseline", "TemporalFusionTransformer", "DeepAR", "NBeats", "NHiTS",
            "DecoderMLP", "RecurrentNetwork", "LSTM".
        data_path (str): Path to the CSV file.
        max_epochs (int): Number of training epochs.
        batch_size (int): Batch size.
    """
    # Read the CSV file and pre-process
    data = pd.read_csv(data_path, header=0)
    data["YearMonth"] = pd.to_datetime(data["YearMonth"], format="%Y%m")
    data["time_idx"] = data["YearMonth"].dt.year * 12 + data["YearMonth"].dt.month
    data["time_idx"] -= data["time_idx"].min()
    # Additional features used for models other than NBeats/NHiTS
    data["month"] = data["YearMonth"].dt.month.astype(str).astype("category")
    data["Sales"] = data["Sales"].clip(lower=1e-8)
    data["log_sales"] = np.log(data["Sales"])
    data["avg_sales_by_sku"] = data.groupby(["time_idx", "SKU"], observed=True)["Sales"].transform("mean")
    data["avg_sales_by_agency"] = data.groupby(["time_idx", "Agency"], observed=True)["Sales"].transform("mean")
    data["Promotions"] = data["Promotions"].astype(float)
    
    # Define time series parameters
    max_encoder_length = 12   # use the last 12 timesteps for encoding
    max_prediction_length = 1  # forecast one timestep ahead

    # Branch dataset creation depending on model type
    if model_name in ["NBeats", "NHiTS"]:
        # For NBeats and NHiTS, only the target is used and target scaling is disabled.
        training = TimeSeriesDataSet(
            data,
            time_idx="time_idx",
            target="Promotions",
            group_ids=["Agency", "SKU"],
            min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length,
            min_prediction_length=max_prediction_length,
            max_prediction_length=max_prediction_length,
            time_varying_unknown_reals=["Promotions"],
            target_normalizer=GroupNormalizer(groups=["Agency", "SKU"], transformation=None),
            add_target_scales=False,
        )
    else:
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
    
    # Create dataloaders
    num_workers = 16
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
    validation_dataset = TimeSeriesDataSet.from_dataset(training, data, stop_randomization=True)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)
    
    # Set up callbacks and logging
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
    )
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval="step")
    logger = pl.loggers.TensorBoardLogger("lightning_logs", name=model_name)
    
    # Initialize model based on model_name
    if model_name == "Baseline":
        model = Baseline.from_dataset(training)
        future_data = TimeSeriesDataSet.from_dataset(training, data, predict=True)
        predictions = model.predict(future_data)
        print("Predicted Promotions:")
        print(predictions)
        return predictions

    elif model_name == "TemporalFusionTransformer":
        model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.06,
            hidden_size=32,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=7,  # default for quantile predictions
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
    elif model_name == "DeepAR":
        model = DeepAR.from_dataset(
            training,
            learning_rate=0.01,
            hidden_size=16,
            rnn_layers=2,
            dropout=0.1,
            loss=NormalDistributionLoss(),
        )
    elif model_name == "NBeats":
        model = NBeats.from_dataset(
            training,
            learning_rate=1e-3,
            weight_decay=1e-2,
            widths=[32, 512],
            num_blocks=[3, 3],
            num_block_layers=[3, 3],
            backcast_loss_ratio=0.0,
            loss=QuantileLoss(),
        )
    elif model_name == "NHiTS":
        model = NHiTS.from_dataset(
            training,
            learning_rate=1e-3,
            weight_decay=1e-2,
            loss=QuantileLoss(),
        )
    elif model_name == "DecoderMLP":
        model = DecoderMLP.from_dataset(
            training,
            learning_rate=1e-3,
            hidden_size=16,
            dropout=0.1,
            loss=QuantileLoss(),
        )
    elif model_name == "RecurrentNetwork":
        model = RecurrentNetwork.from_dataset(
            training,
            learning_rate=1e-3,
            hidden_size=16,
            rnn_layers=2,
            dropout=0.1,
            loss=MAE(),
        )
    elif model_name.upper() in ["LSTM", "SIMPLELSTM"]:
        sample_batch = next(iter(train_dataloader))
        if isinstance(sample_batch, tuple):
            sample_batch = sample_batch[0]
        input_size = sample_batch["encoder_cont"].shape[-1]
        model = SimpleLSTM(input_size=input_size, hidden_size=16, num_layers=1, lr=0.03)
    else:
        raise ValueError(f"Unsupported model: {model_name}.")

    # Initialize trainer (using GPU if available)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[early_stop_callback, lr_logger],
        logger=logger,
        num_sanity_val_steps=0,
    )
    
    # Train the model
    trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)
    
    # Create a future dataset for predictions and predict
    future_data = TimeSeriesDataSet.from_dataset(training, data, predict=True)
    future_dataloader = future_data.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)
    
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in future_dataloader:
            if isinstance(batch, tuple):
                batch = batch[0]
            # Use "encoder_cont" if available; otherwise fall back to "encoder_target"
            if "encoder_cont" in batch:
                x = batch["encoder_cont"]
            elif "encoder_target" in batch:
                x = batch["encoder_target"]
            else:
                raise KeyError("Neither 'encoder_cont' nor 'encoder_target' found in batch.")
            preds = model(x)
            predictions.append(preds)
    predictions = torch.cat(predictions, dim=0)
    
    print("Predicted Promotions:")
    print(predictions)
    return predictions

def main():
    # Map abbreviations to full model names
    abbr_mapping = {
        "TFT": "TemporalFusionTransformer",
        # "DAR": "DeepAR",
        "NBE": "NBeats",
        "NHI": "NHiTS",
        "LSTM": "LSTM",
        # "DMLP": "DecoderMLP",
        # "RN": "RecurrentNetwork",
        # "BL": "Baseline"
    }
    
    parser = argparse.ArgumentParser(description="Run a PyTorch Forecasting model on a CSV dataset.")
    parser.add_argument("--data_path", type=str,
                        default=r"C:\Users\domdd\Documents\GitHub\SportsAnalysis\data\price_sales_promotion.csv",
                        help="Path to the CSV data file")
    parser.add_argument("--model", type=str, default="LSTM", 
                        help=("Model abbreviation to use. Options are: TFT, DAR, NBE, NHI, LSTM, DMLP, RN, BL"))
    parser.add_argument("--max_epochs", type=int, default=1, help="Number of training epochs") 
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training") 
    args = parser.parse_args()
    model_full_name = abbr_mapping.get(args.model.upper(), args.model)
    
    print(f"Running model: {model_full_name} on data: {args.data_path}")
    run_forecasting_model(model_full_name, args.data_path, max_epochs=args.max_epochs, batch_size=args.batch_size)

if __name__ == '__main__':
    main()
