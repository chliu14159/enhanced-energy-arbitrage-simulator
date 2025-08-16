import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class EnergyPricePredictor:
    """Comprehensive neural network framework for energy price prediction"""
    
    def __init__(self, sequence_length=24, prediction_horizon=1):
        """
        Initialize the predictor
        
        Args:
            sequence_length: Number of time steps to look back (24 = 6 hours of 15-min data)
            prediction_horizon: Number of steps to predict ahead
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        self.models = {}
        self.history = {}
        self.predictions = {}
        self.metrics = {}
        
    def load_and_prepare_data(self, data_path='cleaned_data/energy_data_cleaned.csv'):
        """Load and prepare the energy data"""
        print("Loading and preparing data...")
        
        # Load data
        self.df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"Loaded dataset shape: {self.df.shape}")
        
        # Select features based on EDA insights
        self.feature_columns = [
            '日前出清电价',      # Day-ahead price (strongest predictor)
            '新能源预测',        # Renewable forecast (strong negative correlation)
            '竞价空间(火电)',    # Thermal bidding space (strong positive correlation)
            '负荷预测',          # Load forecast
            '风电预测',          # Wind forecast  
            '光伏预测',          # Solar forecast
            '水电',              # Hydro generation
            'hour',              # Hour of day
            'day_of_week',       # Day of week
            'is_peak',           # Peak hour indicator
            'is_weekend'         # Weekend indicator
        ]
        
        self.target_column = '实时出清电价'  # Real-time price
        
        # Create feature matrix
        self.X = self.df[self.feature_columns].values
        self.y = self.df[self.target_column].values.reshape(-1, 1)
        
        print(f"Features shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")
        
        # Scale the data
        self.X_scaled = self.scaler_features.fit_transform(self.X)
        self.y_scaled = self.scaler_target.fit_transform(self.y)
        
        return self.X_scaled, self.y_scaled
    
    def create_sequences(self, X, y):
        """Create sequences for time series prediction"""
        X_seq, y_seq = [], []
        
        for i in range(self.sequence_length, len(X) - self.prediction_horizon + 1):
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i:i+self.prediction_horizon])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train_test_split(self, test_size=0.2, val_size=0.1):
        """Create time-based train/validation/test splits"""
        print("Creating time-based train/val/test splits...")
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(self.X_scaled, self.y_scaled)
        
        # Time-based splits (no random shuffling for time series)
        n_samples = len(X_seq)
        test_start = int(n_samples * (1 - test_size))
        val_start = int(n_samples * (1 - test_size - val_size))
        
        self.X_train = X_seq[:val_start]
        self.y_train = y_seq[:val_start]
        
        self.X_val = X_seq[val_start:test_start]
        self.y_val = y_seq[val_start:test_start]
        
        self.X_test = X_seq[test_start:]
        self.y_test = y_seq[test_start:]
        
        print(f"Train shape: X={self.X_train.shape}, y={self.y_train.shape}")
        print(f"Val shape: X={self.X_val.shape}, y={self.y_val.shape}")
        print(f"Test shape: X={self.X_test.shape}, y={self.y_test.shape}")
        
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
    
    def build_lstm_model(self, units=[64, 32], dropout=0.2, learning_rate=0.001):
        """Build LSTM model"""
        print("Building LSTM model...")
        
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, len(self.feature_columns))),
            layers.LSTM(units[0], return_sequences=True, dropout=dropout),
            layers.LSTM(units[1], dropout=dropout),
            layers.Dense(32, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(self.prediction_horizon)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.models['LSTM'] = model
        return model
    
    def build_cnn_model(self, filters=[64, 32], kernel_size=3, learning_rate=0.001):
        """Build CNN model for temporal pattern recognition"""
        print("Building CNN model...")
        
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, len(self.feature_columns))),
            layers.Conv1D(filters[0], kernel_size, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Conv1D(filters[1], kernel_size, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.prediction_horizon)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.models['CNN'] = model
        return model
    
    def build_gru_model(self, units=[64, 32], dropout=0.2, learning_rate=0.001):
        """Build GRU model (simpler alternative to LSTM)"""
        print("Building GRU model...")
        
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, len(self.feature_columns))),
            layers.GRU(units[0], return_sequences=True, dropout=dropout),
            layers.GRU(units[1], dropout=dropout),
            layers.Dense(32, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(self.prediction_horizon)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.models['GRU'] = model
        return model
    
    def build_transformer_model(self, d_model=64, num_heads=4, ff_dim=128, learning_rate=0.001):
        """Build Transformer model with self-attention"""
        print("Building Transformer model...")
        
        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
            # Multi-head self-attention
            attention_layer = layers.MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout
            )
            attention_output = attention_layer(inputs, inputs)
            attention_output = layers.Dropout(dropout)(attention_output)
            attention_output = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
            
            # Feed-forward network
            ffn = keras.Sequential([
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(inputs.shape[-1]),
            ])
            ffn_output = ffn(attention_output)
            ffn_output = layers.Dropout(dropout)(ffn_output)
            return layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
        
        # Build model
        inputs = layers.Input(shape=(self.sequence_length, len(self.feature_columns)))
        x = transformer_encoder(inputs, head_size=d_model//num_heads, num_heads=num_heads, ff_dim=ff_dim, dropout=0.1)
        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(self.prediction_horizon)(x)
        
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.models['Transformer'] = model
        return model
    
    def train_model(self, model_name, epochs=100, batch_size=32, early_stopping_patience=15):
        """Train a specific model"""
        print(f"Training {model_name} model...")
        
        model = self.models[model_name]
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=early_stopping_patience, 
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=10, 
                min_lr=1e-6
            )
        ]
        
        # Train model
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history[model_name] = history
        return history
    
    def evaluate_model(self, model_name):
        """Evaluate model performance"""
        print(f"Evaluating {model_name} model...")
        
        model = self.models[model_name]
        
        # Get predictions
        y_pred_scaled = model.predict(self.X_test)
        
        # Inverse transform predictions
        y_pred = self.scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1))
        y_true = self.scaler_target.inverse_transform(self.y_test.reshape(-1, 1))
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'MAE': mae,
            'MSE': mse, 
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape
        }
        
        self.metrics[model_name] = metrics
        self.predictions[model_name] = {'y_true': y_true, 'y_pred': y_pred}
        
        print(f"{model_name} Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n=== MODEL COMPARISON ===")
        
        comparison_df = pd.DataFrame(self.metrics).T
        comparison_df = comparison_df.round(4)
        
        print("\nMetrics Comparison:")
        print(comparison_df)
        
        # Find best model for each metric
        print("\nBest Models by Metric:")
        for metric in comparison_df.columns:
            if metric in ['MAE', 'MSE', 'RMSE', 'MAPE']:
                best_model = comparison_df[metric].idxmin()
                best_value = comparison_df.loc[best_model, metric]
                print(f"  {metric}: {best_model} ({best_value:.4f})")
            else:  # R²
                best_model = comparison_df[metric].idxmax()
                best_value = comparison_df.loc[best_model, metric]
                print(f"  {metric}: {best_model} ({best_value:.4f})")
        
        return comparison_df
    
    def plot_results(self, save_plots=True):
        """Create comprehensive visualization of results"""
        print("Creating result visualizations...")
        
        n_models = len(self.models)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        for i, model_name in enumerate(self.models.keys()):
            # Plot training history
            if model_name in self.history:
                history = self.history[model_name]
                axes[0, i].plot(history.history['loss'], label='Train Loss')
                axes[0, i].plot(history.history['val_loss'], label='Val Loss')
                axes[0, i].set_title(f'{model_name} Training History')
                axes[0, i].set_xlabel('Epoch')
                axes[0, i].set_ylabel('Loss')
                axes[0, i].legend()
                axes[0, i].grid(True)
            
            # Plot predictions vs actual
            if model_name in self.predictions:
                pred_data = self.predictions[model_name]
                y_true = pred_data['y_true'][:200]  # Show first 200 predictions
                y_pred = pred_data['y_pred'][:200]
                
                axes[1, i].plot(y_true, label='Actual', alpha=0.7)
                axes[1, i].plot(y_pred, label='Predicted', alpha=0.7)
                axes[1, i].set_title(f'{model_name} Predictions vs Actual')
                axes[1, i].set_xlabel('Time Step')
                axes[1, i].set_ylabel('Price (RMB/MWh)')
                axes[1, i].legend()
                axes[1, i].grid(True)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create metrics comparison plot
        if len(self.metrics) > 1:
            comparison_df = pd.DataFrame(self.metrics).T
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            metrics = ['MAE', 'MSE', 'RMSE', 'R²', 'MAPE']
            for i, metric in enumerate(metrics):
                if metric in comparison_df.columns:
                    bars = axes[i].bar(comparison_df.index, comparison_df[metric])
                    axes[i].set_title(f'{metric} Comparison')
                    axes[i].set_ylabel(metric)
                    axes[i].tick_params(axis='x', rotation=45)
                    
                    # Highlight best model
                    if metric in ['MAE', 'MSE', 'RMSE', 'MAPE']:
                        best_idx = comparison_df[metric].idxmin()
                    else:
                        best_idx = comparison_df[metric].idxmax()
                    
                    for j, bar in enumerate(bars):
                        if comparison_df.index[j] == best_idx:
                            bar.set_color('green')
                        else:
                            bar.set_color('lightblue')
            
            axes[5].axis('off')  # Hide the last subplot
            plt.tight_layout()
            if save_plots:
                plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def run_complete_experiment(self):
        """Run complete modeling experiment"""
        print("Starting complete neural network modeling experiment...")
        print("="*60)
        
        # 1. Load and prepare data
        self.load_and_prepare_data()
        
        # 2. Create train/val/test splits
        self.train_test_split()
        
        # 3. Build all models
        self.build_lstm_model()
        self.build_cnn_model() 
        self.build_gru_model()
        self.build_transformer_model()
        
        # 4. Train all models
        for model_name in self.models.keys():
            print(f"\n{'-'*40}")
            self.train_model(model_name, epochs=50, batch_size=32)
            
        # 5. Evaluate all models
        print(f"\n{'-'*40}")
        print("EVALUATION PHASE")
        print(f"{'-'*40}")
        
        for model_name in self.models.keys():
            self.evaluate_model(model_name)
            print()
        
        # 6. Compare models
        comparison_df = self.compare_models()
        
        # 7. Create visualizations
        self.plot_results()
        
        print(f"\n{'-'*40}")
        print("EXPERIMENT COMPLETED!")
        print(f"{'-'*40}")
        
        return comparison_df

def main():
    """Main function to run the neural network experiment"""
    
    # Initialize predictor
    predictor = EnergyPricePredictor(sequence_length=24, prediction_horizon=1)
    
    # Run complete experiment
    results = predictor.run_complete_experiment()
    
    # Save results
    results.to_csv('model_comparison_results.csv')
    print("\nResults saved to 'model_comparison_results.csv'")
    
    return predictor, results

if __name__ == "__main__":
    predictor, results = main()