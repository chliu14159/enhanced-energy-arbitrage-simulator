#!/usr/bin/env python3
"""
Production Renewable Energy Forecasting Service
==============================================

This script provides a production-ready forecasting service for renewable energy generation
using the trained LSTM models.

Features:
- Load trained models and scalers
- Real-time forecasting for wind and solar generation
- Batch forecasting capabilities
- Model performance monitoring
- Integration-ready API

Author: GitHub Copilot Assistant
Date: 22 August 2025
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
import json

warnings.filterwarnings('ignore')

class ProductionRenewableForecastingService:
    """Production service for renewable energy forecasting."""
    
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scalers = {}
        self.feature_scalers = {}
        self.metadata = {}
        
        # Configuration
        self.sequence_length = 60  # 1 hour of minute-level data
        self.forecast_horizon = 15  # Forecast 15 minutes ahead
        
        self._load_models_and_scalers()
        
        print(f"🔮 Production Renewable Forecasting Service Ready")
        print(f"📊 Loaded models for stations: {list(self.models.keys())}")
    
    def _load_models_and_scalers(self):
        """Load all trained models and scalers."""
        try:
            # Load scalers
            scalers_path = self.models_dir / "scalers.pkl"
            with open(scalers_path, 'rb') as f:
                scaler_data = pickle.load(f)
                self.scalers = scaler_data['target_scalers']
                self.feature_scalers = scaler_data['feature_scalers']
            
            # Load metadata
            metadata_path = self.models_dir / "model_metadata.pkl"
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            # Load models
            for station_id in self.metadata.keys():
                model_path = self.models_dir / f"best_model_{station_id}.h5"
                if model_path.exists():
                    from tensorflow.keras.models import load_model
                    self.models[station_id] = load_model(str(model_path))
                    print(f"✅ Loaded model for station {station_id}")
                else:
                    print(f"⚠️ Model not found for station {station_id}")
        
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def create_time_features(self, dt):
        """Create time-based features for a given datetime."""
        features = {}
        
        # Cyclical time features
        features['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
        features['minute_sin'] = np.sin(2 * np.pi * dt.minute / 60)
        features['minute_cos'] = np.cos(2 * np.pi * dt.minute / 60)
        
        # Day of week and month features
        features['day_of_week'] = dt.weekday()
        features['month'] = dt.month
        features['day_of_year'] = dt.timetuple().tm_yday
        
        # Cyclical day and month features
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        return features
    
    def forecast_station(self, station_id: str, recent_data: pd.DataFrame, forecast_time: datetime):
        """
        Make a forecast for a specific station.
        
        Args:
            station_id: Station identifier
            recent_data: DataFrame with recent generation data (last 60+ minutes)
            forecast_time: Datetime for which to make forecast
        
        Returns:
            dict: Forecast result with value, confidence, and metadata
        """
        if station_id not in self.models:
            return {
                'error': f'No model available for station {station_id}',
                'station_id': station_id,
                'forecast_time': forecast_time.isoformat()
            }
        
        try:
            # Get station metadata
            station_type = self.metadata[station_id]['station_type']
            feature_cols = self.metadata[station_id]['feature_cols']
            
            # Prepare recent data
            recent_data = recent_data.sort_values('datetime')
            
            # Create time features for recent data
            enhanced_data = recent_data.copy()
            for idx, row in enhanced_data.iterrows():
                dt = pd.to_datetime(row['datetime'])
                time_features = self.create_time_features(dt)
                for key, value in time_features.items():
                    enhanced_data.at[idx, key] = value
            
            # Calculate lag and rolling features
            enhanced_data['value_lag1'] = enhanced_data['value'].shift(1)
            enhanced_data['value_lag5'] = enhanced_data['value'].shift(5)
            enhanced_data['value_lag15'] = enhanced_data['value'].shift(15)
            enhanced_data['value_rolling_mean_15'] = enhanced_data['value'].rolling(15).mean()
            enhanced_data['value_rolling_std_15'] = enhanced_data['value'].rolling(15).std()
            enhanced_data['value_rolling_mean_60'] = enhanced_data['value'].rolling(60).mean()
            
            # Add solar-specific features if needed
            if station_type == 'solar':
                for idx, row in enhanced_data.iterrows():
                    dt = pd.to_datetime(row['datetime'])
                    hour_angle = (dt.hour - 12) * 15  # Degrees from noon
                    sun_elevation = max(0, np.cos(np.radians(hour_angle)) * np.cos(np.radians(30)))
                    enhanced_data.at[idx, 'sun_elevation_proxy'] = sun_elevation
            
            # Remove NaN rows
            enhanced_data = enhanced_data.dropna()
            
            # Check if we have enough data
            if len(enhanced_data) < self.sequence_length:
                return {
                    'error': f'Insufficient data: need {self.sequence_length} records, got {len(enhanced_data)}',
                    'station_id': station_id,
                    'forecast_time': forecast_time.isoformat()
                }
            
            # Extract features for the last sequence
            feature_data = enhanced_data[feature_cols].values[-self.sequence_length:]
            
            # Scale features
            feature_scaler = self.feature_scalers[station_id]
            scaled_features = feature_scaler.transform(feature_data)
            
            # Reshape for model input
            model_input = scaled_features.reshape(1, self.sequence_length, len(feature_cols))
            
            # Make prediction
            model = self.models[station_id]
            scaled_prediction = model.predict(model_input, verbose=0)
            
            # Inverse transform prediction
            target_scaler = self.scalers[station_id]
            prediction = target_scaler.inverse_transform(scaled_prediction.reshape(-1, 1))[0, 0]
            
            # Ensure non-negative values
            prediction = max(0, prediction)
            
            # Calculate confidence (simplified approach)
            recent_std = enhanced_data['value'].std()
            model_mae = self.metadata[station_id]['metrics']['mae']
            confidence = max(0, 1 - (model_mae / (recent_std + 1e-8)))
            
            return {
                'station_id': station_id,
                'station_type': station_type,
                'forecast_time': forecast_time.isoformat(),
                'predicted_generation': float(prediction),
                'confidence': float(confidence),
                'model_mae': float(model_mae),
                'recent_avg': float(enhanced_data['value'].tail(15).mean()),
                'recent_std': float(recent_std)
            }
        
        except Exception as e:
            return {
                'error': f'Forecasting failed: {str(e)}',
                'station_id': station_id,
                'forecast_time': forecast_time.isoformat()
            }
    
    def forecast_all_stations(self, recent_data_dict: dict, forecast_time: datetime):
        """
        Make forecasts for all available stations.
        
        Args:
            recent_data_dict: Dict mapping station_id to recent data DataFrame
            forecast_time: Datetime for which to make forecasts
        
        Returns:
            dict: Forecasts for all stations
        """
        forecasts = {}
        
        for station_id in self.models.keys():
            if station_id in recent_data_dict:
                forecast = self.forecast_station(
                    station_id, 
                    recent_data_dict[station_id], 
                    forecast_time
                )
                forecasts[station_id] = forecast
            else:
                forecasts[station_id] = {
                    'error': f'No recent data provided for station {station_id}',
                    'station_id': station_id,
                    'forecast_time': forecast_time.isoformat()
                }
        
        return forecasts
    
    def get_model_performance(self):
        """Get performance metrics for all models."""
        performance = {}
        
        for station_id, metadata in self.metadata.items():
            performance[station_id] = {
                'station_type': metadata['station_type'],
                'metrics': metadata['metrics'],
                'feature_count': len(metadata['feature_cols']),
                'model_available': station_id in self.models
            }
        
        return performance
    
    def simulate_forecast_from_data(self, data_path: str, forecast_minutes: int = 15):
        """
        Simulate forecasting using historical data for validation.
        
        Args:
            data_path: Path to historical data
            forecast_minutes: Minutes ahead to forecast
        
        Returns:
            dict: Simulation results
        """
        # Load historical data
        if Path(data_path).suffix == '.parquet':
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        results = {}
        
        for station_id in self.models.keys():
            station_data = df[df['station_id'] == station_id].sort_values('datetime')
            
            if len(station_data) < 100:  # Need enough data for simulation
                continue
            
            # Use last part of data for simulation
            test_start = len(station_data) - 100
            recent_data = station_data.iloc[:test_start + self.sequence_length]
            actual_future = station_data.iloc[test_start + self.sequence_length:]
            
            if len(actual_future) == 0:
                continue
            
            # Make forecast for first future point
            forecast_time = actual_future.iloc[0]['datetime']
            forecast_result = self.forecast_station(station_id, recent_data, forecast_time)
            
            if 'predicted_generation' in forecast_result:
                actual_value = actual_future.iloc[0]['value']
                error = abs(forecast_result['predicted_generation'] - actual_value)
                relative_error = error / (actual_value + 1e-8) * 100
                
                results[station_id] = {
                    'forecast': forecast_result,
                    'actual_value': float(actual_value),
                    'absolute_error': float(error),
                    'relative_error': float(relative_error)
                }
            else:
                results[station_id] = {'error': forecast_result.get('error', 'Unknown error')}
        
        return results


def demo_forecasting():
    """Demonstrate the forecasting service."""
    print("🎯 RENEWABLE ENERGY FORECASTING DEMO")
    print("=" * 50)
    
    # Initialize service
    models_dir = "/Users/randomwalk/Documents/CODE/REPO/energy_trading_js/models/lstm_forecasting"
    service = ProductionRenewableForecastingService(models_dir)
    
    # Show model performance
    print("\n📊 Model Performance Summary:")
    performance = service.get_model_performance()
    for station_id, perf in performance.items():
        if perf['model_available']:
            print(f"  Station {station_id} ({perf['station_type']}):")
            print(f"    MAE: {perf['metrics']['mae']:.1f} MW")
            print(f"    MAPE: {perf['metrics']['mape']:.1f}%")
            print(f"    R²: {perf['metrics']['r2']:.3f}")
    
    # Simulate forecasting with historical data
    print("\n🔮 Simulation with Historical Data:")
    data_path = "/Users/randomwalk/Documents/CODE/REPO/energy_trading_js/processed/wind_solar/wind_solar_data_cleaned_20250822_200203.parquet"
    
    simulation_results = service.simulate_forecast_from_data(data_path)
    
    for station_id, result in simulation_results.items():
        if 'forecast' in result:
            print(f"\n🏭 Station {station_id}:")
            print(f"  Predicted: {result['forecast']['predicted_generation']:.1f} MW")
            print(f"  Actual: {result['actual_value']:.1f} MW")
            print(f"  Error: {result['absolute_error']:.1f} MW ({result['relative_error']:.1f}%)")
            print(f"  Confidence: {result['forecast']['confidence']:.3f}")
    
    # Example of how to use for real-time forecasting
    print("\n🚀 Production Usage Example:")
    print("""
# For real-time forecasting:
service = ProductionRenewableForecastingService(models_dir)

# Prepare recent data (last 60+ minutes)
recent_data = get_recent_station_data(station_id)  # Your data source

# Make forecast
forecast_time = datetime.now() + timedelta(minutes=15)
forecast = service.forecast_station(station_id, recent_data, forecast_time)

print(f"Forecast for {forecast_time}: {forecast['predicted_generation']:.1f} MW")
""")
    
    print("\n✅ Forecasting Demo Complete!")


if __name__ == "__main__":
    demo_forecasting()
