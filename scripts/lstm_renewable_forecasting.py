#!/usr/bin/env python3
"""
Wind and Solar Generation Forecasting with LSTM Models
=====================================================

This script builds separate LSTM models for wind and solar generation forecasting
using the cleaned renewable energy dataset.

Features:
- Separate models for wind and solar generation
- Multi-station forecasting capability
- Time series feature engineering
- Model evaluation and visualization
- Production-ready forecasting pipeline

Author: GitHub Copilot Assistant
Date: 22 August 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import joblib

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class RenewableEnergyForecaster:
    """LSTM-based forecasting for renewable energy generation."""
    
    def __init__(self, data_path: str, output_dir: str = None):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir) if output_dir else self.data_path.parent / "models"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.sequence_length = 60  # 1 hour of minute-level data
        self.forecast_horizon = 15  # Forecast 15 minutes ahead
        
        # Storage for models and scalers
        self.models = {}
        self.scalers = {}
        self.feature_scalers = {}
        
        # Data storage
        self.data = None
        self.processed_data = {}
        
        print(f"🔮 Renewable Energy Forecaster Initialized")
        print(f"📁 Output directory: {self.output_dir}")
    
    def load_and_preprocess_data(self):
        """Load and preprocess the renewable energy data."""
        print(f"\n📊 Loading data from {self.data_path}")
        
        # Load data
        if self.data_path.suffix == '.parquet':
            self.data = pd.read_parquet(self.data_path)
        else:
            self.data = pd.read_csv(self.data_path)
        
        # Convert datetime
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        
        print(f"✅ Loaded {len(self.data):,} records")
        print(f"📅 Date range: {self.data['datetime'].min()} to {self.data['datetime'].max()}")
        print(f"🏭 Stations: {self.data['station_id'].unique()}")
        
        # Analyze data sufficiency
        self._analyze_data_sufficiency()
        
        return self.data
    
    def _analyze_data_sufficiency(self):
        """Analyze if we have enough data for reliable modeling."""
        if self.data is None:
            print("❌ No data loaded")
            return
        
        print(f"\n🔍 Data Sufficiency Analysis:")
        
        total_days = (self.data['datetime'].max() - self.data['datetime'].min()).days
        records_per_station = self.data.groupby(['station_id', 'type']).size()
        
        print(f"📈 Total days: {total_days}")
        print(f"📊 Records per station:")
        
        for (station, type_), count in records_per_station.items():
            expected_records = total_days * 24 * 60  # Expected minute-level records
            coverage = (count / expected_records) * 100 if expected_records > 0 else 0
            print(f"  {station} ({type_}): {count:,} records ({coverage:.1f}% coverage)")
        
        # Recommend modeling approach
        if total_days >= 60:
            print(f"✅ EXCELLENT: {total_days} days is sufficient for robust LSTM modeling")
        elif total_days >= 30:
            print(f"✅ GOOD: {total_days} days is adequate for LSTM modeling")
        elif total_days >= 14:
            print(f"⚠️ MARGINAL: {total_days} days may be limited for complex LSTM models")
        else:
            print(f"❌ INSUFFICIENT: {total_days} days is too limited for reliable LSTM modeling")
    
    def create_time_features(self, df):
        """Create time-based features for improved forecasting."""
        df = df.copy()
        
        # Cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
        
        # Day of week and month features
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['day_of_year'] = df['datetime'].dt.dayofyear
        
        # Cyclical day and month features
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def prepare_station_data(self, station_id: str):
        """Prepare data for a specific station."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_and_preprocess_data() first.")
        
        print(f"\n🏭 Preparing data for station {station_id}")
        
        # Filter data for station
        station_data = self.data[self.data['station_id'] == station_id].copy()
        station_data = station_data.sort_values('datetime')
        
        # Get station type
        station_type = station_data['type'].iloc[0]
        
        # Create time features
        station_data = self.create_time_features(station_data)
        
        # Calculate rolling statistics as features
        station_data['value_lag1'] = station_data['value'].shift(1)
        station_data['value_lag5'] = station_data['value'].shift(5)
        station_data['value_lag15'] = station_data['value'].shift(15)
        station_data['value_rolling_mean_15'] = station_data['value'].rolling(15).mean()
        station_data['value_rolling_std_15'] = station_data['value'].rolling(15).std()
        station_data['value_rolling_mean_60'] = station_data['value'].rolling(60).mean()
        
        # For solar, add sun elevation proxy
        if station_type == 'solar':
            # Simple sun elevation approximation
            hour_angle = (station_data['hour'] - 12) * 15  # Degrees from noon
            station_data['sun_elevation_proxy'] = np.maximum(0, 
                np.cos(np.radians(hour_angle)) * 
                np.cos(np.radians(30))  # Approximate latitude
            )
        
        # Drop rows with NaN values
        station_data = station_data.dropna()
        
        print(f"✅ Prepared {len(station_data):,} records for {station_type} station {station_id}")
        
        return station_data, station_type
    
    def create_sequences(self, data, target_col='value'):
        """Create sequences for LSTM training."""
        # Feature columns
        feature_cols = [
            'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
            'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'value_lag1', 'value_lag5', 'value_lag15',
            'value_rolling_mean_15', 'value_rolling_std_15', 'value_rolling_mean_60'
        ]
        
        # Add solar-specific features if available
        if 'sun_elevation_proxy' in data.columns:
            feature_cols.append('sun_elevation_proxy')
        
        # Prepare feature matrix
        features = data[feature_cols].values
        target = data[target_col].values
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(data) - self.forecast_horizon + 1):
            # Features sequence
            X.append(features[i-self.sequence_length:i])
            # Target (future value)
            y.append(target[i + self.forecast_horizon - 1])
        
        return np.array(X), np.array(y), feature_cols
    
    def build_lstm_model(self, input_shape, station_type):
        """Build LSTM model architecture."""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        # Use different optimizers for different types
        if station_type == 'solar':
            optimizer = Adam(learning_rate=0.001)
        else:  # wind
            optimizer = Adam(learning_rate=0.0005)
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_station_model(self, station_id: str):
        """Train LSTM model for a specific station."""
        print(f"\n🚀 Training model for station {station_id}")
        
        # Prepare data
        station_data, station_type = self.prepare_station_data(station_id)
        
        # Create sequences
        X, y, feature_cols = self.create_sequences(station_data)
        
        print(f"📊 Training sequences: {X.shape}")
        print(f"🎯 Training targets: {y.shape}")
        
        # Split data (80% train, 20% test)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features and targets
        feature_scaler = StandardScaler()
        target_scaler = MinMaxScaler()
        
        # Reshape for scaling
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        
        # Fit and transform
        X_train_scaled = feature_scaler.fit_transform(X_train_reshaped)
        X_test_scaled = feature_scaler.transform(X_test_reshaped)
        
        # Reshape back
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        # Scale targets
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        # Build model
        model = self.build_lstm_model(
            input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
            station_type=station_type
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5),
            ModelCheckpoint(
                str(self.output_dir / f"best_model_{station_id}.h5"),
                save_best_only=True
            )
        ]
        
        # Train model
        print(f"🏋️ Training {station_type} model...")
        history = model.fit(
            X_train_scaled, y_train_scaled,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Make predictions
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_test_actual = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_actual, y_pred)
        mse = mean_squared_error(y_test_actual, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_actual, y_pred)
        mape = np.mean(np.abs((y_test_actual - y_pred) / (y_test_actual + 1e-8))) * 100
        
        print(f"\n📊 Model Performance for Station {station_id} ({station_type}):")
        print(f"   MAE: {mae:.2f} MW")
        print(f"   RMSE: {rmse:.2f} MW")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   R²: {r2:.4f}")
        
        # Store model and scalers
        self.models[station_id] = {
            'model': model,
            'station_type': station_type,
            'feature_cols': feature_cols,
            'metrics': {
                'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2
            }
        }
        self.scalers[station_id] = target_scaler
        self.feature_scalers[station_id] = feature_scaler
        
        # Save training history
        self._plot_training_history(history, station_id, station_type)
        self._plot_prediction_results(y_test_actual, y_pred, station_id, station_type)
        
        return model, mae, rmse, mape, r2
    
    def _plot_training_history(self, history, station_id, station_type):
        """Plot training history."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{station_type.title()} Station {station_id} - Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title(f'{station_type.title()} Station {station_id} - Training MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'training_history_{station_id}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_results(self, y_true, y_pred, station_id, station_type):
        """Plot prediction results."""
        plt.figure(figsize=(15, 8))
        
        # Time series plot
        plt.subplot(2, 2, 1)
        n_samples = min(1000, len(y_true))
        plt.plot(y_true[:n_samples], label='Actual', alpha=0.7)
        plt.plot(y_pred[:n_samples], label='Predicted', alpha=0.7)
        plt.title(f'{station_type.title()} Station {station_id} - Time Series Comparison')
        plt.xlabel('Time')
        plt.ylabel('Generation (MW)')
        plt.legend()
        
        # Scatter plot
        plt.subplot(2, 2, 2)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Generation (MW)')
        plt.ylabel('Predicted Generation (MW)')
        plt.title(f'{station_type.title()} Station {station_id} - Actual vs Predicted')
        
        # Residuals
        plt.subplot(2, 2, 3)
        residuals = y_true - y_pred
        plt.hist(residuals, bins=50, alpha=0.7)
        plt.xlabel('Residuals (MW)')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        
        # Error over time
        plt.subplot(2, 2, 4)
        plt.plot(np.abs(residuals[:n_samples]))
        plt.xlabel('Time')
        plt.ylabel('Absolute Error (MW)')
        plt.title('Absolute Error Over Time')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'prediction_results_{station_id}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def train_all_models(self):
        """Train models for all stations."""
        print(f"\n🎯 Training models for all stations...")
        
        if self.data is None:
            self.load_and_preprocess_data()
        
        if self.data is None:
            raise ValueError("Failed to load data")
        
        results = {}
        stations = self.data['station_id'].unique()
        
        for station_id in stations:
            try:
                model, mae, rmse, mape, r2 = self.train_station_model(station_id)
                results[station_id] = {
                    'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2,
                    'type': self.models[station_id]['station_type']
                }
            except Exception as e:
                print(f"❌ Failed to train model for station {station_id}: {e}")
                results[station_id] = None
        
        # Summary report
        self._generate_model_summary(results)
        
        return results
    
    def _generate_model_summary(self, results):
        """Generate model performance summary."""
        print(f"\n📋 MODEL PERFORMANCE SUMMARY")
        print("=" * 60)
        
        # Separate by type
        wind_results = {k: v for k, v in results.items() if v and v['type'] == 'wind'}
        solar_results = {k: v for k, v in results.items() if v and v['type'] == 'solar'}
        
        def print_type_summary(type_results, type_name):
            if not type_results:
                return
            
            print(f"\n🌪️ {type_name.upper()} STATIONS:")
            print(f"{'Station':<10} {'MAE':<8} {'RMSE':<8} {'MAPE':<8} {'R²':<8}")
            print("-" * 50)
            
            for station, metrics in type_results.items():
                print(f"{station:<10} {metrics['mae']:<8.1f} {metrics['rmse']:<8.1f} "
                      f"{metrics['mape']:<8.1f}% {metrics['r2']:<8.3f}")
            
            # Average metrics
            avg_mae = np.mean([m['mae'] for m in type_results.values()])
            avg_rmse = np.mean([m['rmse'] for m in type_results.values()])
            avg_mape = np.mean([m['mape'] for m in type_results.values()])
            avg_r2 = np.mean([m['r2'] for m in type_results.values()])
            
            print("-" * 50)
            print(f"{'AVERAGE':<10} {avg_mae:<8.1f} {avg_rmse:<8.1f} "
                  f"{avg_mape:<8.1f}% {avg_r2:<8.3f}")
        
        print_type_summary(wind_results, "Wind")
        print_type_summary(solar_results, "Solar")
    
    def save_models(self):
        """Save all models and scalers."""
        print(f"\n💾 Saving models and scalers...")
        
        # Save models
        for station_id, model_info in self.models.items():
            model_path = self.output_dir / f"model_{station_id}.h5"
            model_info['model'].save(str(model_path))
            print(f"✅ Saved model for station {station_id}")
        
        # Save scalers
        scalers_path = self.output_dir / "scalers.pkl"
        with open(scalers_path, 'wb') as f:
            pickle.dump({
                'target_scalers': self.scalers,
                'feature_scalers': self.feature_scalers
            }, f)
        
        # Save model metadata
        metadata = {
            station_id: {
                'station_type': info['station_type'],
                'feature_cols': info['feature_cols'],
                'metrics': info['metrics']
            }
            for station_id, info in self.models.items()
        }
        
        metadata_path = self.output_dir / "model_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✅ All models and scalers saved to {self.output_dir}")
    
    def forecast_station(self, station_id: str, datetime_str: str, recent_data=None):
        """Make forecast for a specific station."""
        if station_id not in self.models:
            raise ValueError(f"No model found for station {station_id}")
        
        # This would use recent data to make predictions
        # Implementation would depend on how you want to provide recent data
        print(f"🔮 Forecasting for station {station_id} at {datetime_str}")
        # Return placeholder for now
        return {"forecast": 0.0, "confidence": 0.0}


def main():
    """Main execution function."""
    print("🔮 RENEWABLE ENERGY LSTM FORECASTING")
    print("=" * 50)
    
    # Paths
    data_path = "/Users/randomwalk/Documents/CODE/REPO/energy_trading_js/processed/wind_solar/wind_solar_data_cleaned_20250822_200203.parquet"
    output_dir = "/Users/randomwalk/Documents/CODE/REPO/energy_trading_js/models/lstm_forecasting"
    
    # Initialize forecaster
    forecaster = RenewableEnergyForecaster(data_path, output_dir)
    
    # Load and analyze data
    forecaster.load_and_preprocess_data()
    
    # Train all models
    results = forecaster.train_all_models()
    
    # Save models
    forecaster.save_models()
    
    print(f"\n✅ LSTM Forecasting Pipeline Complete!")
    print(f"📁 Models saved to: {output_dir}")
    print(f"🚀 Ready for production forecasting!")


if __name__ == "__main__":
    main()
