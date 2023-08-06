import os 
import sys 

import torch
import torch.nn as nn
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import plotly.io as pio
import optuna.visualization as vis
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from abc import ABC, abstractmethod


class ForecastingModel(ABC):
    DEFAULT_CONFIG={}
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


    @abstractmethod
    def fit(self, params, epochs=100, patience=10):
        pass


    @abstractmethod
    def predict(self, x):
        pass


    def parse_config(self, config):
        parsed_config = {}
        for key, default in self.DEFAULT_CONFIG.items():
            if key in config:
                parsed_config[key] = config[key]
            else:
                parsed_config[key] = default

        return parsed_config

    def update_data(self, x_train, y_train, x_test, y_test):
        """
        Update the training and testing data.

        :param x_train: The new training features.
        :param y_train: The new training targets.
        :param x_test: The new testing features.
        :param y_test: The new testing targets.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def plot_loss(self, prefix):
        """
        Saves the training and validation loss curves to a PDF.

        :param prefix: The prefix for the PDF filename.
        """
        with PdfPages(f'{prefix}_loss_curves.pdf') as pdf:
            # Create a plot with matplotlib for the loss curves
            fig, ax = plt.subplots()
            ax.plot(self.train_losses, label='Train Loss')
            ax.plot(self.val_losses, label='Validation Loss')
            ax.set_title('Train and Validation Loss Curves')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()

            # Save the matplotlib figure
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


class Seq2SeqLSTM(ForecastingModel):
    DEFAULT_CONFIG = {
        'num_layers': (1, 3),
        'hidden_size': (16, 128),
        'dropout': (0, 0.5),
        'lr': (1e-5, 1e-1),
        'weight_decay': (1e-5, 1e-1),
        'l1_lambda': (1e-9, 1e-3),
        'epochs': 100,
        'patience': 10 
    }
    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__(x_train, y_train, x_test, y_test)
        self.input_size = x_train.shape[-1]
        self.output_size = y_train.shape[-1]
        
        self.study=None
        self.best_params=None


    def _objective(self, trial, config):
        parsed_config = self.parse_config(config)

        # Extract or suggest hyperparameters based on the parsed_config
        num_layers = trial.suggest_int('num_layers', *parsed_config['num_layers'], log=False)
        hidden_size = trial.suggest_int('hidden_size', *parsed_config['hidden_size'], log=False)
        dropout = trial.suggest_float('dropout', *parsed_config['dropout'], log=False)
        learning_rate = trial.suggest_float('lr', *parsed_config['lr'], log=True)
        weight_decay = trial.suggest_float('weight_decay', *parsed_config['weight_decay'], log=False)
        l1_lambda = trial.suggest_float('l1_lambda', *parsed_config['l1_lambda'], log=False)
        epochs = parsed_config['epochs']
        patience = parsed_config['patience']

        model = LSTMModel(input_size=self.input_size, 
                        hidden_size=hidden_size, 
                        num_layers=num_layers, 
                        output_size=self.output_size, 
                        dropout=dropout)

        train_losses, val_losses = model.fit(
            self.x_train, 
            self.y_train, 
            self.x_test, 
            self.y_test, 
            learning_rate=learning_rate, 
            weight_decay=weight_decay,
            l1_lambda=l1_lambda, 
            epochs=epochs,
            patience=patience)

        # store the loss history in the trial's user attributes
        trial.set_user_attr('train_losses', train_losses)
        trial.set_user_attr('val_losses', val_losses)

        # return the final validation loss
        return val_losses[-1]


    def predict(self, x):
        """
        Predict the target variable using the trained model.

        :param x: The features to predict for.
        :return: The model's predictions.
        """
        # Put the model in evaluation mode
        self.model.eval()

        # Predict
        with torch.no_grad():
            predictions = self.model(x)

        return predictions


    def fit(self, params, epochs=100, patience=20):
        """
        Fit the model using the given hyperparameters.

        :param params: The hyperparameters to use when fitting the model. This should be a dictionary with the same keys as DEFAULT_CONFIG.
        """

        # Extract hyperparameters from the input
        num_layers = params['num_layers']
        hidden_size = params['hidden_size']
        dropout = params['dropout']
        learning_rate = params['lr']
        weight_decay = params['weight_decay']
        l1_lambda = params['l1_lambda']

        # Instantiate the model with the given hyperparameters
        self.model = LSTMModel(input_size=self.input_size, 
                        hidden_size=hidden_size, 
                        num_layers=num_layers, 
                        output_size=self.output_size, 
                        dropout=dropout)

        # Fit the model to the data
        self.train_losses, self.val_losses = self.model.fit(
            self.x_train, 
            self.y_train, 
            self.x_test, 
            self.y_test, 
            learning_rate=learning_rate, 
            weight_decay=weight_decay,
            l1_lambda=l1_lambda, 
            epochs=epochs,
            patience=patience)
        
        # return the final validation loss
        return self.val_losses[-1]


    def optimize(self, n_trials=100, config={}, warmup_steps=10, pruner=True):
        if pruner:
            pruner = MedianPruner(n_warmup_steps=warmup_steps)
        else:
            pruner = None
        sampler = TPESampler(n_startup_trials=warmup_steps)

        self.study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
        self.study.optimize(lambda trial: self._objective(trial, config), n_trials=n_trials)

        # Print results
        print('Number of finished trials:', len(self.study.trials))
        print('Best trial:')
        trial = self.study.best_trial
        print('Value: ', trial.value)
        print('Params: ')
        for key, value in trial.params.items():
            print(f'    {key}: {value}')

        self.best_params = trial.params
        return self.best_params


    def plot_trials(self, prefix):
        # Plot Optuna's built-in summary plots
        fig = optuna.visualization.plot_optimization_history(self.study)
        pio.write_image(fig, f'{prefix}_optimization_history.pdf')

        fig = optuna.visualization.plot_parallel_coordinate(self.study)
        pio.write_image(fig, f'{prefix}_parallel_coordinate.pdf')

        fig = optuna.visualization.plot_slice(self.study)
        pio.write_image(fig, f'{prefix}_slice.pdf')

        fig = optuna.visualization.plot_contour(self.study)
        pio.write_image(fig, f'{prefix}_contour.pdf')

        # Plot loss curves for each trial
        with PdfPages(f'{prefix}_loss_curves.pdf') as pdf:
            # Sort trials by their value in ascending order
            sorted_trials = sorted(self.study.trials, key=lambda t: t.value)
            for trial in sorted_trials:
                train_losses = trial.user_attrs['train_losses']
                val_losses = trial.user_attrs['val_losses']

                # Create a plot with matplotlib for the loss curves
                fig, ax = plt.subplots()
                ax.plot(train_losses, label='Train Loss')
                ax.plot(val_losses, label='Validation Loss')
                ax.set_title(f'Trial {trial.number}: Value={trial.value:.4f}, Params={trial.params}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                
                # Save the matplotlib figure
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)


    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)


    def load_model(self, file_path):
        self.model = LSTMModel(input_size=self.input_size, 
                        hidden_size=self.hidden_size, 
                        num_layers=self.num_layers, 
                        output_size=self.output_size, 
                        dropout=self.dropout)
        self.model.load_state_dict(torch.load(file_path))


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


    def fit(self, x_train, y_train, x_val, y_val, learning_rate, weight_decay, l1_lambda, epochs, patience=10):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        train_losses = []
        val_losses = []

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            outputs = self(x_train)
            train_loss = criterion(outputs, y_train)

            # Adding L1 regularization
            l1_reg = torch.tensor(0.)
            for param in self.parameters():
                l1_reg += torch.norm(param, 1)
            
            train_loss += l1_lambda * l1_reg  

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())

            with torch.no_grad():
                val_outputs = self(x_val)
                val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())

            # check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # early stopping
            if epochs_no_improve == patience:
                #print(f"Early stopping at epoch {epoch}")
                break

        return train_losses, val_losses
