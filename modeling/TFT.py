import copy
from pathlib import Path
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

# Read the CSV file
data = pd.read_csv(r"C:\Users\domdd\Documents\GitHub\SportsAnalysis\data\price_sales_promotion.csv")

# Convert YearMonth (in "YYYYMM" format) to datetime
data["YearMonth"] = pd.to_datetime(data["YearMonth"], format="%Y%m")

# Create a time index from the YearMonth column
data["time_idx"] = data["YearMonth"].dt.year * 12 + data["YearMonth"].dt.month
data["time_idx"] -= data["time_idx"].min()

# Create additional features
data["month"] = data["YearMonth"].dt.month.astype(str).astype("category")

# Ensure Sales has no zeros or negative values by clipping to a small positive number
data["Sales"] = data["Sales"].clip(lower=1e-8)

# Compute a log transformation of Sales
data["log_sales"] = np.log(data["Sales"])

# Compute average Sales by SKU and Agency per time index
data["avg_sales_by_sku"] = data.groupby(["time_idx", "SKU"], observed=True)["Sales"].transform("mean")
data["avg_sales_by_agency"] = data.groupby(["time_idx", "Agency"], observed=True)["Sales"].transform("mean")

# Define encoder and prediction lengths
max_encoder_length = 12   # use the last 12 time steps for encoding
max_prediction_length = 1  # forecast the next time step

# Create a TimeSeriesDataSet for training
training = TimeSeriesDataSet(
    data,
    time_idx="time_idx",
    target="Promotions",
    group_ids=["Agency", "SKU"],
    min_encoder_length=max_encoder_length,  # require at least 12 past time steps
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    # Known time-varying real features available in both past and future
    time_varying_known_reals=["time_idx", "Price", "Sales", "log_sales", "avg_sales_by_sku", "avg_sales_by_agency"],
    # Known time-varying categorical features
    time_varying_known_categoricals=["month"],
    # The target variable ("Promotions") is unknown in the future
    time_varying_unknown_reals=["Promotions"],
    target_normalizer=GroupNormalizer(
        groups=["Agency", "SKU"],
        transformation="softplus"
    ),
    add_relative_time_idx=True,  # adds a relative time index feature
    add_target_scales=True,      # scales the target
    add_encoder_length=True,     # adds the encoder length as a feature
)

# Create a DataLoader for training
batch_size = 32  # adjust based on your dataset size and system capabilities
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)

# Create a validation dataset and corresponding dataloader
validation = TimeSeriesDataSet.from_dataset(training, data, stop_randomization=True)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

# Set up callbacks and logging for training
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor(logging_interval="step")
logger = TensorBoardLogger("lightning_logs", name="tft_promotions")

# Create the Temporal Fusion Transformer model from the training dataset
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.06,
    hidden_size=32,           # adjust as needed
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,            # default output size for predictions
    loss=MAE(),             # using Mean Absolute Error loss instead of QuantileLoss
    log_interval=10,
    reduce_on_plateau_patience=4,
)

# Override log_prediction to disable prediction plot logging (avoids negative yerr issue)
tft.log_prediction = lambda *args, **kwargs: {}

# Initialize the PyTorch Lightning Trainer
trainer = pl.Trainer(
    max_epochs=5,
    accelerator="gpu",  # change to "gpu" if you have a GPU available
    devices=1,          # set to the number of GPUs you want to use (e.g., 1)
    callbacks=[early_stop_callback, lr_logger],
    logger=logger,
    num_sanity_val_steps=0  # disable sanity validation steps
)

# Train the model with the validation dataloader
trainer.fit(tft, train_dataloader, val_dataloaders=val_dataloader)

# Create a future dataframe for forecasting the next time step
future_data = TimeSeriesDataSet.from_dataset(training, data, predict=True)

# Make predictions using the trained model
predictions = tft.predict(future_data)

print("Predicted Promotions:")
print(predictions)
