import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

# Define a simple LSTM model using PyTorch Lightning
class SimpleLSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size=16, num_layers=1, lr=0.03):
        super().__init__()
        self.lr = lr
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Forecast a single value

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
    
    def _get_input(self, batch):
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
