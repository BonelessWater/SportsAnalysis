import argparse
import pandas as pd
import numpy as np
import torch
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

import os
import warnings

# Environment variables
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", message="Attribute 'loss' is an instance of `nn.Module`")

# -----------------------------------------------------------
# Override the TFT plotting to avoid negative errorbars error
# -----------------------------------------------------------
class MyTFT(TemporalFusionTransformer):
    def plot_prediction(self, x, out, idx=None, add_loss_to_title=False, **kwargs):
        # Instead of plotting predictions (which triggers error due to negative error bars),
        # simply return None so that logging doesn't try to plot.
        return None

# -----------------------------------------------------------
# Evaluation function (unchanged)
# -----------------------------------------------------------
def evaluate_dataloader(model, dataloader, model_name, prediction_length):
    """
    Evaluate the given dataloader using the provided model.
    Returns flattened numpy arrays for predictions and targets.
    """
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, tuple):
                batch = batch[0]
            if model_name in ["Baseline", "DeepAR", "RecurrentNetwork"] or "encoder_lengths" in batch:
                preds = model(batch)
            else:
                if "encoder_cont" in batch:
                    x = batch["encoder_cont"]
                elif "encoder_target" in batch:
                    x = batch["encoder_target"]
                else:
                    raise KeyError("No appropriate input found in batch.")
                preds = model(x)
            if not isinstance(preds, torch.Tensor) and hasattr(preds, "prediction"):
                preds = preds.prediction

            # If predictions have an extra sample dimension, aggregate it.
            if preds.ndim == 3:
                if preds.shape[1] == prediction_length:
                    preds = preds.mean(axis=-1)
                else:
                    preds = preds.mean(axis=1)
            all_preds.append(preds.cpu().numpy())

            if "decoder_target" in batch:
                target_array = batch["decoder_target"].cpu().numpy()
            elif "target" in batch:
                target_array = batch["target"].cpu().numpy()
                if target_array.ndim > 1 and target_array.shape[1] != prediction_length:
                    target_array = target_array[:, -prediction_length:]
            else:
                raise KeyError("No target found in batch.")
            all_targets.append(target_array)
            
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    return all_preds, all_targets

# -----------------------------------------------------------
# Team mapping: full NBA team names to their three-letter tickers
# -----------------------------------------------------------
TEAM_NAME_TO_TICKER = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "New Jersey Nets": "NJN",  # For 2004-2011
    "Charlotte Bobcats": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Hornets": "NOH",
    "New York Knicks": "NYK",
    "Seattle SuperSonics": "SEA",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS"
}

def run_forecasting_model(model_name: str, data_path: str, max_epochs: int = 5, batch_size: int = 32, full_data: bool = False):
    """
    Runs the specified forecasting model on the NBA dataset.
    The CSV (nba_data.csv) contains game-level features and aggregated player statistics.
    """
    # Read the CSV file with index_col=0 (if that's how it was saved)
    data = pd.read_csv(data_path, index_col=0)
    
    # Convert game_date (stored as YYYYMMDD integer) to datetime
    data["game_date"] = pd.to_datetime(data["game_date"].astype(str), format="%Y%m%d")
    
    # Create a new column for the home team ticker using the mapping
    data["home_team_ticker"] = data["home_team"].apply(lambda x: TEAM_NAME_TO_TICKER.get(x, x))
    data["home_team_ticker"] = data["home_team_ticker"].astype("category")
    
    # Sort data by team and game_date, then create a sequential time index for each team
    data = data.sort_values(["home_team_ticker", "game_date"]).reset_index(drop=True)
    data["time_idx"] = data.groupby("home_team_ticker").cumcount() + 1
    
    # Compute days from the first game for each team
    data["days_from_start"] = data.groupby("home_team_ticker")["game_date"].transform(lambda x: (x - x.min()).dt.days)
    
    # Ensure the season_year column exists; if not, create a default one.
    if "season_year" not in data.columns:
        print("Warning: 'season_year' column not found in CSV. Creating a default season_year column.")
        data["season_year"] = 0
    # Convert season_year to numeric (do not convert to categorical)
    data["season_year"] = pd.to_numeric(data["season_year"], errors="coerce")
    
    # Convert target to float so that softplus transformation works
    data["target"] = data["target"].astype(np.float32)
    
    # Optionally, subset the data for faster testing (e.g., take only the first 10 games per team)
    if not full_data:
        data = data.groupby("home_team_ticker").head(10)
        print("Using a subset of the data for faster testing.")
    
    # Set forecasting horizons: use last 6 games for the encoder and predict the next game (prediction_length = 1)
    max_encoder_length = 6
    max_prediction_length = 1
    
    # Create the TimeSeriesDataSet for the NBA data.
    # We use the aggregated features from home and visitor players as known real features.
    training = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="target",  # target is home_pts - visitor_pts
        group_ids=["home_team_ticker"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        # Known reals: time index, days from start, season_year, and some aggregated player features.
        time_varying_known_reals=[
            "time_idx", "days_from_start", "season_year",
            "home_avg_age", "home_avg_height", "home_avg_exp",
            "visitor_avg_age", "visitor_avg_height", "visitor_avg_exp"
        ],
        # No categorical known reals (season_year is numeric now)
        time_varying_known_categoricals=[],
        # Unknown reals: the target variable
        time_varying_unknown_reals=["target"],
        target_normalizer=GroupNormalizer(
            groups=["home_team_ticker"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # Create dataloaders for training and validation
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    validation_dataset = TimeSeriesDataSet.from_dataset(training, data, stop_randomization=True)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    
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
        print("Baseline Predictions:")
        print(predictions)
    elif model_name == "TemporalFusionTransformer":
        model = MyTFT.from_dataset(
            training,
            learning_rate=0.06,
            hidden_size=32,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=7,  # default for quantile predictions
            loss=QuantileLoss(),
            log_interval=0,
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
    
    # Save the model checkpoint or state_dict
    if model_name != "Baseline":
        checkpoint_path = f"../outputs/saved_models/{model_name}_model.ckpt"
        trainer.save_checkpoint(checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
    else:
        torch.save(model.state_dict(), f"../outputs/saved_models/{model_name}_model.pt")
        print(f"Baseline model saved to {model_name}_model.pt")
    
    # Evaluate performance on the training set
    train_preds, train_targets = evaluate_dataloader(model, train_dataloader, model_name, max_prediction_length)
    train_mae = mean_absolute_error(train_targets, train_preds)
    train_mse = mean_squared_error(train_targets, train_preds)
    train_rmse = np.sqrt(train_mse)
    data_sorted = data.sort_values("time_idx")
    if len(data_sorted["target"].values) > 1:
        naive_error = np.mean(np.abs(np.diff(data_sorted["target"].values)))
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
    
    # Evaluate on the validation set
    test_preds, test_targets = evaluate_dataloader(model, val_dataloader, model_name, max_prediction_length)
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
        "LSTM": "LSTM",
        "DAR": "DeepAR",
        "DMLP": "DecoderMLP",
        "RN": "RecurrentNetwork",
        "NBE": "NBeats",
        "NHI": "NHiTS",
        "BL": "Baseline"
    }
    
    parser = argparse.ArgumentParser(description="Run a PyTorch Forecasting model on an NBA CSV dataset.")
    parser.add_argument("--data_path", type=str,
                        default="../outputs/data/nba/nba_data.csv",
                        help="Path to the CSV data file")
    parser.add_argument("--model", type=str, default="LSTM", 
                        help=("Model abbreviation to use. Options are: TFT, DAR, LSTM, DMLP, RN, NBE, NHI, BL"))
    parser.add_argument("--max_epochs", type=int, default=1, help="Number of training epochs") 
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training") 
    parser.add_argument("--full_data", action="store_true", 
                        help="If set, use the full dataset; otherwise, only a subset is loaded for faster testing")
    args = parser.parse_args()
    model_full_name = abbr_mapping.get(args.model.upper(), args.model)
    
    print(f"Running model: {model_full_name} on data: {args.data_path}")
    preds, metrics = run_forecasting_model(model_full_name, args.data_path, max_epochs=args.max_epochs, batch_size=args.batch_size, full_data=args.full_data)
    print("\nFinal Performance Metrics:")
    for phase in metrics:
        print(f"\n{phase} Metrics:")
        for k, v in metrics[phase].items():
            print(f"{k}: {v}")

if __name__ == '__main__':
    main()
