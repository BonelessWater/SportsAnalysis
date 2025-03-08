import warnings
import pandas as pd
import numpy as np
import lightning.pytorch as pl

from pytorch_forecasting import TimeSeriesDataSet, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer

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
    
    # Create dataloaders
    batch_size = 32
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    validation = TimeSeriesDataSet.from_dataset(training, data, stop_randomization=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    
    # Instantiate DeepAR model (if available)
    try:
        from pytorch_forecasting.models.deepar import DeepAR
        deepar_model = DeepAR.from_dataset(
            training,
            learning_rate=0.03,
            loss=QuantileLoss(),
            log_interval=10,
        )
    except ImportError:
        print("DeepAR model is not available in your PyTorch Forecasting version.")
        return
    
    # Set up trainer and train
    trainer = pl.Trainer(max_epochs=1, accelerator="gpu", devices=1, num_sanity_val_steps=0)
    trainer.fit(deepar_model, train_dataloader, val_dataloaders=val_dataloader)
    
    # Make predictions
    future_data = TimeSeriesDataSet.from_dataset(training, data, predict=True)
    predictions = deepar_model.predict(future_data)
    print("DeepAR Model Predictions:")
    print(predictions)

if __name__ == "__main__":
    main()
