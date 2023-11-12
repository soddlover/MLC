import torch
import torch.nn as nn
#impoting lightning:
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import preprocessing
import numpy as np
import pandas as pd
import torch.nn.functional as F
#import dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as DataSet
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import preprocessing as pp
import postprocessing as pop
import random
from pytorch_lightning import seed_everything

SEED = 69
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.use_deterministic_algorithms(True)
g = torch.Generator()
g.manual_seed(SEED)



def seed_worker(worker_id):
    worker_seed = SEED
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class SolarForecastingDataset(DataSet):
    def __init__(self, features_df, target_series):
        """
        Initializes the dataset with features and target labels.

        :param features_df: DataFrame containing the features.
        :param target_series: Series containing the target labels.
        """
        self.features = features_df
        self.targets = target_series

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        # Extracting the features and the target label for the given index
        feature_vector = self.features.iloc[index].values
        target_label = self.targets.iloc[index]

        return {
            "feature_vector": torch.tensor(feature_vector, dtype=torch.float),
            "target_label": torch.tensor(target_label, dtype=torch.float)
        }

class SolarForecastingDatasetDataModule(pl.LightningDataModule):
    def __init__(self, train_features_df, train_targets_series, test_features_df, test_targets_series, batch_size=8):
        super().__init__()
        self.train_features_df = train_features_df
        self.train_targets_series = train_targets_series
        self.test_features_df = test_features_df
        self.test_targets_series = test_targets_series
        self.batch_size = batch_size

        
    def setup(self, stage=None):
        self.train_dataset = SolarForecastingDataset(self.train_features_df, self.train_targets_series)
        self.test_dataset = SolarForecastingDataset(self.test_features_df, self.test_targets_series)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,worker_init_fn=seed_worker, generator=g, shuffle=True,)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, worker_init_fn=seed_worker, generator=g)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, worker_init_fn=seed_worker, generator=g)

class FullyConnectedDNN(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, dropout_prob=0.1):
        super(FullyConnectedDNN, self).__init__()
        # Create fully connected layers
        self.fc_layers = nn.ModuleList()
        for i in range(len(layer_sizes)):
            in_features = input_size if i == 0 else layer_sizes[i - 1]
            out_features = layer_sizes[i]
            self.fc_layers.append(nn.Linear(in_features, out_features))

        self.output_layer = nn.Linear(layer_sizes[-1], output_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        for layer in self.fc_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x

def weighted_mae_loss(input, target, exponent=1, constant=1):
    assert input.size() == target.size()

    # Calculate the absolute error
    absolute_errors = torch.abs(input - target)

    # Apply exponential scaling with a constant
    adjusted_target = target + constant
    weighted_errors = absolute_errors * (adjusted_target ** exponent)

    return weighted_errors.mean()

class CustomMAELoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(CustomMAELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets):
        errors = torch.abs(inputs - targets)  # Compute the absolute errors
        capped_errors = torch.clamp(errors, max=10000)
        loss = self.alpha * torch.pow(capped_errors, self.beta) 
        return torch.mean(loss)  # Return the mean loss


class SolarPowerProductionPredictor(pl.LightningModule):

    def __init__(self, input_size, layer_sizes, output_size, weight_decay=1e-5, dropout_prob=0.1, learning_rate=0.01, verbose=True, loss_exponent=1.0, loss_beta=1.0):
        super().__init__()
        self.model = FullyConnectedDNN(input_size, layer_sizes, output_size, dropout_prob=dropout_prob)
        self.criterion = self.criterion = lambda input, target: weighted_mae_loss(input, target, exponent=loss_exponent, constant=1)
        #self.criterion = CustomMAELoss(loss_alpha, loss_beta)

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.verbose = verbose
    
    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        features, labels = batch["feature_vector"], batch["target_label"]
        loss, outputs = self(features, labels) 
        self.log("train_loss", loss, prog_bar=self.verbose, logger=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        features, labels = batch["feature_vector"], batch["target_label"]
        loss, outputs = self(features, labels) 
        self.log("val_loss", loss, prog_bar=self.verbose, logger=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        features, labels = batch["feature_vector"], batch["target_label"]
        loss, outputs = self(features, labels) 
        self.log("test_loss", loss, prog_bar=self.verbose, logger=False)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)




class CustomModelCheckpoint(ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Save the best model path to the pl_module
        pl_module.best_model_path = self.best_model_path

def get_predictions(model, dataloader):
    model.eval()  # set the model to evaluation mode
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            features, labels = batch["feature_vector"], batch["target_label"]
            predictions = model(features)[1]  
            if not isinstance(predictions, torch.Tensor):
                raise TypeError("Model output is not a tensor. Got type: {}".format(type(predictions)))
            
            all_predictions.append(predictions)
            all_labels.append(labels)

    # Check for tensor types before concatenation
    if not all(isinstance(p, torch.Tensor) for p in all_predictions):
        raise TypeError("Not all elements in predictions are tensors.")

    if not all(isinstance(l, torch.Tensor) for l in all_labels):
        raise TypeError("Not all elements in labels are tensors.")

    all_predictions_tensor = torch.cat(all_predictions, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)

    # Convert tensors to numpy arrays
    all_predictions_np = all_predictions_tensor.cpu().numpy()
    all_labels_np = all_labels_tensor.cpu().numpy()
    
    return all_predictions_np, all_labels_np

class HenrikDNN:

    def __init__(self,n_features = None, layer_sizes = [100,50], output_size = 1, drop_out_prob = 0.1, learning_rate = 0.01, weight_decay = 1e-5, max_epochs = 100, paitience = 5, batch_size = 16, val_chack_interval = 1, pruning_callback = None, verbose = True, loss_expontent = 1):

        self.n_features = n_features
        self.layer_sizes = layer_sizes
        self.verbose = verbose
        self.output_size = output_size
        self.drop_out_prob = drop_out_prob
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_expontent = loss_expontent
        self.paitience = paitience
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.val_chack_interval = val_chack_interval
        self.pruning_callback = pruning_callback
        SEED = 69
        print("hei")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        torch.use_deterministic_algorithms(True)
        g = torch.Generator()
        g.manual_seed(SEED)


        self.pl_model = SolarPowerProductionPredictor(self.n_features, self.layer_sizes, self.output_size, weight_decay=self.weight_decay, dropout_prob=self.drop_out_prob, learning_rate=self.learning_rate, verbose=self.verbose, loss_exponent=self.loss_expontent)

        self.checkpoint_callback = CustomModelCheckpoint(
            dirpath='HenrikDNN_checkpoints',
            save_top_k=1,
            verbose=self.verbose,
            monitor='val_loss',
            mode='min',
            filename='model-{epoch:02d}-{val_loss:.2f}'
        )

        self.early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            patience=self.paitience
        )


        self.callbacks = [self.early_stopping_callback, self.checkpoint_callback]
        if self.pruning_callback is not None:
            self.callbacks.append(self.pruning_callback)
        seed_everything(69, workers=True)

        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=self.callbacks,
            enable_progress_bar=self.verbose,
            accelerator="cpu",
            check_val_every_n_epoch = 2,
            deterministic=True

            #val_check_interval=self.val_chack_interval
        )

    def train(self, X_train, y_train, X_val, y_val):
        """
        Args:
            X_train: Training df with datetime index. 
            y_train: Training df, with datetime index. Each row in y_train corresponds to four rows in X_train.
    
        """
        #print the seed:
        SEED = 69
        print("hei")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        torch.use_deterministic_algorithms(True)
        g = torch.Generator()
        g.manual_seed(SEED)

        self.data_module = SolarForecastingDatasetDataModule(X_train, y_train, X_val, y_val, batch_size=self.batch_size)
        self.trainer.fit(self.pl_model, self.data_module)


    def predict(self, X):
        trained_model = SolarPowerProductionPredictor.load_from_checkpoint(
            self.pl_model.best_model_path,
            input_size=self.n_features,
            layer_sizes=self.layer_sizes,
            output_size=self.output_size,
            dropout_prob=self.drop_out_prob,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            verbose=self.verbose,
            loss_exponent=self.loss_expontent
        )

        X_dataloader = torch.utils.data.DataLoader(
            SolarForecastingDataset(X, pd.Series(np.zeros(X.shape[0]))),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=g
        )

        predictions, _ = get_predictions(trained_model, X_dataloader)

        return predictions



def train_test_split_on_specific_day_May_june(X, y, split_date):
    """
    Splits the data based on a given date. Additionally, moves May, June and July data of split_date's year
    from training set to test set.
    
    Parameters:
    - X: Quarter-hourly input data with DateTime index.
    - y: Hourly target data with DateTime index.
    - split_date: Date (string or datetime object) to split the data on.
    
    Returns:
    X_train, y_train, X_test, y_test
    """
    split_date = pd.Timestamp(split_date).normalize()

    # Ensure split_date is a datetime object
    if isinstance(split_date, str):
        split_date = pd.Timestamp(split_date)

    print(f"Split date: {split_date}")

    # Split the data based on the provided date
    X_train = X[X.index.normalize() < split_date]
    y_train = y[y.index.normalize() < split_date]

    X_test = X[X.index.normalize() >= split_date]
    y_test = y[y.index.normalize() >= split_date]

    # Define conditions to move May and June of split_date's year from train to test
    may_june_condition_X = ((X_train.index.month == 5) | (X_train.index.month == 6) | (X_train.index.month == 7)) & (X_train.index.year == split_date.year)
    may_june_condition_y = ((y_train.index.month == 5) | (y_train.index.month == 6) | (y_train.index.month == 7)) & (y_train.index.year == split_date.year)
    
    X_may_june = X_train[may_june_condition_X]
    y_may_june = y_train[may_june_condition_y]

    # Remove May and June data from training set
    X_train = X_train[~may_june_condition_X]
    y_train = y_train[~may_june_condition_y]

    # Append May and June data to test set
    X_test = pd.concat([X_may_june, X_test])
    y_test = pd.concat([y_may_june, y_test])

    return X_train, y_train, X_test, y_test


#tests:
"---------------------------------------------------------------------------------------------------------------------"


if __name__ == "__main__":
    print("Loading data...")
    X,y = preprocessing.general_read("A")
    X = preprocessing.concatenate_dfs(X)
    #X, y = plot_and_remove_days(X, y)
    #X.drop(columns=['uncertainty'], inplace=True)
    X = preprocessing.QuartersAsColumnsTransformer().transform(X)



    X_train, y_train, X_val, y_val = pp.train_test_split_may_june_july(X, y, "A")

    y_train = y_train["target"]
    y_val = y_val["target"]

    target_encoder = preprocessing.HourMonthTargetEncoder()
    target_encoder.fit(X_train, y_train)
    X_train = target_encoder.transform(X_train)
    X_val = target_encoder.transform(X_val)

    y_train = pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_val)
    # y_val_weights = pd.DataFrame(y_val_weights)



    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    y_scaler.fit(y_train)
    X_scaler.fit(X_train)

    X_train = pd.DataFrame(
        X_scaler.transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )

    X_val = pd.DataFrame(
        X_scaler.transform(X_val),
        index=X_val.index,
        columns=X_val.columns
    )

    y_train = pd.DataFrame(
        y_scaler.transform(y_train),
        index=y_train.index,
        columns=y_train.columns
    )

    y_val = pd.DataFrame(
        y_scaler.transform(y_val),
        index=y_val.index,
        columns=y_val.columns
    )
#287.22350817503116	quarters	true	119	101	0.00009088260234731966	0.9702228408589507	2.499126185711371e-8
    params = {
        "n_features": X_train.shape[1],
        "layer_sizes": [119,101],
        "output_size": 1,
        "drop_out_prob": 0.03,
        "learning_rate": 0.00009088260234731966,
        "max_epochs": 200,
        "paitience": 10,
        "batch_size": 16,
        "val_chack_interval": 0.5,
        "weight_decay": 2.499126185711371e-8,
        "loss_expontent": 0.9702228408589507,
        "verbose": True
    }

    # # Initialize the model
    model = HenrikDNN(**params)
    print("Training model...")

    #check for nan:
    model.train(X_train, y_train, X_val, y_val)
    predictions = model.predict(X_val)
    predictions = y_scaler.inverse_transform(predictions)
    y_val = y_scaler.inverse_transform(y_val)

    mae = mean_absolute_error(predictions, y_val)
    print(f"MAE: {mae:.2f}")
    pop.calculate_hourly_mae_and_plot(predictions, y_val)

    print("done")