import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from LSTM import SimpleLSTM
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

# For metrics computations
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

def evaluate_dataloader(model, dataloader, model_name):
    """
    Evaluate the given dataloader using the provided model.
    Returns flattened numpy arrays for predictions and targets.
    """
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # If the batch is a tuple, take the first element (the dictionary)
            if isinstance(batch, tuple):
                batch = batch[0]
            # For Baseline, pass the entire batch
            if model_name == "Baseline":
                preds = model(batch)
            else:
                if "encoder_cont" in batch:
                    x = batch["encoder_cont"]
                elif "encoder_target" in batch:
                    x = batch["encoder_target"]
                else:
                    raise KeyError("No appropriate input found in batch.")
                preds = model(x)
            # If output is not a tensor, extract underlying tensor (e.g. from a 'prediction' attribute)
            if not isinstance(preds, torch.Tensor) and hasattr(preds, "prediction"):
                preds = preds.prediction
            all_preds.append(preds.cpu().numpy())
            # Retrieve targets
            if "decoder_target" in batch:
                all_targets.append(batch["decoder_target"].cpu().numpy())
            elif "target" in batch:
                all_targets.append(batch["target"].cpu().numpy())
            else:
                raise KeyError("No target found in batch.")
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    return all_preds, all_targets

def run_forecasting_model(model_name: str, data_path: str, max_epochs: int = 5, batch_size: int = 2):
    """
    Runs the specified forecasting model on the given dataset.
    After training (or baseline prediction), the model is evaluated on both the training and validation sets.
    Performance metrics (MAE, MSE, RMSE, MASE, and AUC ROC if applicable) are computed and printed.
    The trained model is then saved to disk.
    
    Args:
        model_name (str): Name of the model to use. Options are:
            "Baseline", "TemporalFusionTransformer", "DeepAR", "NBeats", "NHiTS",
            "DecoderMLP", "RecurrentNetwork", "LSTM".
        data_path (str): Path to the CSV file.
        max_epochs (int): Number of training epochs.
        batch_size (int): Batch size.
    """
    # Read the CSV file and pre-process
    data = pd.read_csv(data_path, index_col=0)
    data["date"] = pd.to_datetime(data["date"])
    data["t"] = data["t"].astype(int)
    data["time_idx"] = data["t"] - data["t"].min()

    # Convert categorical columns
    data["month"] = data["month"].astype(str).astype("category")
    data["categorical_day_of_week"] = data["categorical_day_of_week"].astype(str).astype("category")
    data["categorical_hour"] = data["categorical_hour"].astype(str).astype("category")
    data["categorical_id"] = data["categorical_id"].astype(str).astype("category")  # For grouping

    # Process the target variable
    data["power_usage"] = data["power_usage"].clip(lower=1e-8)
    data["log_power_usage"] = np.log(data["power_usage"])
    data["avg_power_usage_by_cat"] = data.groupby(
        ["time_idx", "categorical_id"], observed=True
    )["power_usage"].transform("mean")

    # Define time series parameters
    max_encoder_length = 12   # use the last 12 timesteps for encoding
    max_prediction_length = 1  # forecast one timestep ahead

    # Create dataset and select target column
    if model_name in ["NBeats", "NHiTS"]:
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
        target_col = "Promotions"
    else:
        training = TimeSeriesDataSet(
            data,
            time_idx="t",
            target="power_usage",
            group_ids=["categorical_id"],
            min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length,
            min_prediction_length=max_prediction_length,
            max_prediction_length=max_prediction_length,
            time_varying_known_reals=["t", "days_from_start"],
            time_varying_known_categoricals=["month", "categorical_hour", "categorical_day_of_week"],
            time_varying_unknown_reals=["power_usage"],
            target_normalizer=GroupNormalizer(groups=["categorical_id"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        target_col = "power_usage"

    # Create dataloaders
    num_workers = 16
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
    validation_dataset = TimeSeriesDataSet.from_dataset(training, data, stop_randomization=True)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)
    
    # Set up callbacks and logging (for non-Baseline models)
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
        print("Baseline Predicted Promotions:")
        print(predictions)
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

    # For models other than Baseline, train using PyTorch Lightning
    if model_name != "Baseline":
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=[early_stop_callback, lr_logger],
            logger=logger,
            num_sanity_val_steps=0,
        )
        trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)
    
    # Save the model checkpoint (or state_dict for Baseline)
    if model_name != "Baseline":
        checkpoint_path = f"../outputs/saved_models/{model_name}_model.ckpt"
        trainer.save_checkpoint(checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
    else:
        torch.save(model.state_dict(), f"../outputs/saved_models/{model_name}_model.pt")
        print(f"Baseline model saved to {model_name}_model.pt")
    
    # -------------------------
    # Evaluate performance on the training set
    # -------------------------
    train_preds, train_targets = evaluate_dataloader(model, train_dataloader, model_name)
    train_mae = mean_absolute_error(train_targets, train_preds)
    train_mse = mean_squared_error(train_targets, train_preds)
    train_rmse = np.sqrt(train_mse)
    data_sorted = data.sort_values("t")
    if len(data_sorted[target_col].values) > 1:
        naive_error = np.mean(np.abs(np.diff(data_sorted[target_col].values)))
    else:
        naive_error = np.inf
    train_mase = train_mae / naive_error if naive_error != 0 else float("inf")
    unique_train = np.unique(train_targets)
    if set(unique_train).issubset({0, 1}):
        train_auc = roc_auc_score(train_targets, train_preds)
    else:
        train_auc = None

    print("\nTraining Performance Metrics:")
    print(f"MAE  : {train_mae:.4f}")
    print(f"MSE  : {train_mse:.4f}")
    print(f"RMSE : {train_rmse:.4f}")
    print(f"MASE : {train_mase:.4f}")
    if train_auc is not None:
        print(f"AUC ROC : {train_auc:.4f}")
    else:
        print("AUC ROC : Not applicable for non-binary targets")

    # -------------------------
    # Evaluate performance on the testing (validation) set
    # -------------------------
    test_preds, test_targets = evaluate_dataloader(model, val_dataloader, model_name)
    test_mae = mean_absolute_error(test_targets, test_preds)
    test_mse = mean_squared_error(test_targets, test_preds)
    test_rmse = np.sqrt(test_mse)
    test_mase = test_mae / naive_error if naive_error != 0 else float("inf")
    unique_test = np.unique(test_targets)
    if set(unique_test).issubset({0, 1}):
        test_auc = roc_auc_score(test_targets, test_preds)
    else:
        test_auc = None

    print("\nTesting (Validation) Performance Metrics:")
    print(f"MAE  : {test_mae:.4f}")
    print(f"MSE  : {test_mse:.4f}")
    print(f"RMSE : {test_rmse:.4f}")
    print(f"MASE : {test_mase:.4f}")
    if test_auc is not None:
        print(f"AUC ROC : {test_auc:.4f}")
    else:
        print("AUC ROC : Not applicable for non-binary targets")

    # Combine metrics in a dictionary if needed
    metrics = {
        "Training": {
            "MAE": train_mae,
            "MSE": train_mse,
            "RMSE": train_rmse,
            "MASE": train_mase,
            "AUC_ROC": train_auc,
        },
        "Testing": {
            "MAE": test_mae,
            "MSE": test_mse,
            "RMSE": test_rmse,
            "MASE": test_mase,
            "AUC_ROC": test_auc,
        },
    }
    return test_preds, metrics

def main():
    abbr_mapping = {
        "TFT": "TemporalFusionTransformer",
        "NBE": "NBeats",
        "NHI": "NHiTS",
        "LSTM": "LSTM",
        "RN": "RecurrentNetwork",
        "BL": "Baseline"
    }
    
    parser = argparse.ArgumentParser(description="Run a PyTorch Forecasting model on a CSV dataset.")
    parser.add_argument("--data_path", type=str,
                        default="../outputs/data/electricity/hourly_electricity.csv",
                        help="Path to the CSV data file")
    parser.add_argument("--model", type=str, default="LSTM", 
                        help=("Model abbreviation to use. Options are: TFT, DAR, NBE, NHI, LSTM, DMLP, RN, BL"))
    parser.add_argument("--max_epochs", type=int, default=1, help="Number of training epochs") 
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training") 
    args = parser.parse_args()
    model_full_name = abbr_mapping.get(args.model.upper(), args.model)
    
    print(f"Running model: {model_full_name} on data: {args.data_path}")
    preds, metrics = run_forecasting_model(model_full_name, args.data_path, max_epochs=args.max_epochs, batch_size=args.batch_size)
    print("\nFinal Performance Metrics:")
    for phase in metrics:
        print(f"\n{phase} Metrics:")
        for k, v in metrics[phase].items():
            print(f"{k}: {v}")

if __name__ == '__main__':
    main()
