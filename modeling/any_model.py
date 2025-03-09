import argparse
from pathlib import Path
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import forecasting models
from pytorch_forecasting.models.baseline import Baseline
from pytorch_forecasting.models.deepar import DeepAR
from pytorch_forecasting.models.mlp import DecoderMLP
from pytorch_forecasting.models.nbeats import NBeats
from pytorch_forecasting.models.nhits import NHiTS
from pytorch_forecasting.models.rnn import RecurrentNetwork
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss, MAE

def run_forecasting_model(model_name: str, data_path: str, max_epochs: int = 5, batch_size: int = 2):
    """
    Runs the specified PyTorch Forecasting model on the given dataset.

    Args:
        model_name (str): Name of the model to use. Options are:
            "Baseline", "TemporalFusionTransformer", "DeepAR", "NBeats", "NHiTS",
            "DecoderMLP", "RecurrentNetwork".
        data_path (str): Path to the CSV file.
        max_epochs (int): Number of training epochs.
        batch_size (int): Batch size.
    """
    # Read the CSV file (it should have a header row)
    data = pd.read_csv(data_path, header=0)
    
    # Convert YearMonth (in "YYYYMM") to datetime and create a time index
    data["YearMonth"] = pd.to_datetime(data["YearMonth"], format="%Y%m")
    data["time_idx"] = data["YearMonth"].dt.year * 12 + data["YearMonth"].dt.month
    data["time_idx"] -= data["time_idx"].min()
    
    # Create additional features (used only for models other than DeepAR, NBeats, or NHiTS)
    data["month"] = data["YearMonth"].dt.month.astype(str).astype("category")
    
    # Ensure Sales is positive and compute transformations
    data["Sales"] = data["Sales"].clip(lower=1e-8)
    data["log_sales"] = np.log(data["Sales"])
    data["avg_sales_by_sku"] = data.groupby(["time_idx", "SKU"], observed=True)["Sales"].transform("mean")
    data["avg_sales_by_agency"] = data.groupby(["time_idx", "Agency"], observed=True)["Sales"].transform("mean")
    
    # Ensure the target is numeric
    data["Promotions"] = data["Promotions"].astype(float)
    
    # Define encoder and prediction lengths
    max_encoder_length = 12   # last 12 timesteps for encoding
    max_prediction_length = 1  # forecast one timestep ahead

    # Build dataset depending on model type:
    if model_name == "DeepAR":
        # For DeepAR, we want target scaling enabled so that we have a univariate target.
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
            add_target_scales=True,   # crucial for DeepAR
            add_encoder_length=True,
        )
    elif model_name in ["NBeats", "NHiTS"]:
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
        # For other models, include extra known features.
        known_reals = ["time_idx", "Price", "Sales", "log_sales", "avg_sales_by_sku", "avg_sales_by_agency"]
        known_categoricals = ["month"]
        training = TimeSeriesDataSet(
            data,
            time_idx="time_idx",
            target="Promotions",
            group_ids=["Agency", "SKU"],
            min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length,
            min_prediction_length=max_prediction_length,
            max_prediction_length=max_prediction_length,
            time_varying_known_reals=known_reals,
            time_varying_known_categoricals=known_categoricals,
            time_varying_unknown_reals=["Promotions"],
            target_normalizer=GroupNormalizer(groups=["Agency", "SKU"], transformation=None),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
    
    # Create dataloaders
    num_workers = 16
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
    validation = TimeSeriesDataSet.from_dataset(training, data, stop_randomization=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)
    
    # Set up callbacks and logging
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger("lightning_logs", name=model_name)
    
    # Initialize model based on model_name
    if model_name == "Baseline":
        model = Baseline.from_dataset(training)
        # For Baseline, there's nothing to train, so directly predict.
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
        # DeepAR expects loss as a list with one element.
        model = DeepAR.from_dataset(
            training,
            learning_rate=0.01,
            hidden_size=16,
            rnn_layers=2,
            dropout=0.1,
            loss=[nn.MSELoss()],  # or use another loss wrapped in a list
        )
    elif model_name == "NBeats":
        model = NBeats.from_dataset(
            training,
            learning_rate=1e-3,
            weight_decay=1e-2,
            widths=[32, 32],
            backcast_loss_ratio=0.1,
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
    predictions = model.predict(future_data)
    
    print("Predicted Promotions:")
    print(predictions)
    return predictions

def main():
    # Map abbreviations to full model names
    abbr_mapping = {
        #"BL": "Baseline",
        "TFT": "TemporalFusionTransformer",
        "DAR": "DeepAR",
        "NBE": "NBeats",
        "NHI": "NHiTS",
        #"DMLP": "DecoderMLP",
        #"RN": "RecurrentNetwork",
    }
    
    parser = argparse.ArgumentParser(description="Run a PyTorch Forecasting model on a CSV dataset.")
    parser.add_argument("--data_path", type=str,
                        default=r"C:\Users\domdd\Documents\GitHub\SportsAnalysis\data\price_sales_promotion.csv",
                        help="Path to the CSV data file")
    parser.add_argument("--model", type=str, default="DAR", help=("Model abbreviation to use. Options are: "
                                                                  "BL, TFT, DAR, NBE, NHI, DMLP, RN"))
    parser.add_argument("--max_epochs", type=int, default=1, help="Number of training epochs") 
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training") 
    args = parser.parse_args()
    model_full_name = abbr_mapping.get(args.model.upper(), args.model)
    
    print(f"Running model: {model_full_name} on data: {args.data_path}")
    run_forecasting_model(model_full_name, args.data_path, max_epochs=args.max_epochs, batch_size=args.batch_size)

if __name__ == '__main__':
    main()
