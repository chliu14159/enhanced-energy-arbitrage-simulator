import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

class ImprovedEnergyPredictor:
    """Enhanced neural network models with advanced features and techniques"""
    
    def __init__(self, sequence_length=24, prediction_horizon=1):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler_features = RobustScaler()  # Better for outliers
        self.scaler_target = RobustScaler()
        self.models = {}
        self.results = {}
        
    def advanced_feature_engineering(self, df):
        """Create advanced features for better prediction accuracy"""
        print("Creating advanced features...")
        
        df_enhanced = df.copy()
        target_col = '实时出清电价'
        da_col = '日前出清电价'
        
        # 1. Lag features (key for time series)
        for lag in [1, 2, 4, 8, 24, 48, 96]:  # 15min to 24h lags
            df_enhanced[f'rt_price_lag_{lag}'] = df_enhanced[target_col].shift(lag)
            df_enhanced[f'da_price_lag_{lag}'] = df_enhanced[da_col].shift(lag)
        
        # 2. Moving averages (trend indicators)
        for window in [4, 12, 24, 48]:  # 1h to 12h windows
            df_enhanced[f'rt_price_ma_{window}'] = df_enhanced[target_col].rolling(window).mean()
            df_enhanced[f'rt_price_std_{window}'] = df_enhanced[target_col].rolling(window).std()
            df_enhanced[f'load_ma_{window}'] = df_enhanced['负荷预测'].rolling(window).mean()
            df_enhanced[f'renewable_ma_{window}'] = df_enhanced['新能源预测'].rolling(window).mean()
        
        # 3. Price spread and volatility features
        df_enhanced['price_spread'] = df_enhanced[target_col] - df_enhanced[da_col]
        df_enhanced['price_spread_pct'] = df_enhanced['price_spread'] / df_enhanced[da_col]
        df_enhanced['price_volatility_12h'] = df_enhanced[target_col].rolling(48).std()
        df_enhanced['price_range_12h'] = (df_enhanced[target_col].rolling(48).max() - 
                                         df_enhanced[target_col].rolling(48).min())
        
        # 4. Market balance indicators
        df_enhanced['supply_demand_ratio'] = (df_enhanced['新能源预测'] + df_enhanced['水电'] + 
                                            df_enhanced['非市场化出力']) / df_enhanced['负荷预测']
        df_enhanced['renewable_penetration'] = df_enhanced['新能源预测'] / df_enhanced['负荷预测']
        df_enhanced['thermal_margin'] = df_enhanced['竞价空间(火电)'] / df_enhanced['负荷预测']
        
        # 5. Time-based features (enhanced)
        df_enhanced['hour_sin'] = np.sin(2 * np.pi * df_enhanced['hour'] / 24)
        df_enhanced['hour_cos'] = np.cos(2 * np.pi * df_enhanced['hour'] / 24)
        df_enhanced['day_sin'] = np.sin(2 * np.pi * df_enhanced['day_of_week'] / 7)
        df_enhanced['day_cos'] = np.cos(2 * np.pi * df_enhanced['day_of_week'] / 7)
        df_enhanced['quarter_hour_sin'] = np.sin(2 * np.pi * df_enhanced['quarter_hour'] / 4)
        df_enhanced['quarter_hour_cos'] = np.cos(2 * np.pi * df_enhanced['quarter_hour'] / 4)
        
        # 6. Price regime indicators
        df_enhanced['is_low_price'] = (df_enhanced[target_col] < 50).astype(int)
        df_enhanced['is_high_price'] = (df_enhanced[target_col] > 500).astype(int)
        df_enhanced['is_extreme_volatility'] = (df_enhanced['price_volatility_12h'] > 200).astype(int)
        
        # 7. Interaction features
        df_enhanced['load_renewable_interaction'] = df_enhanced['负荷预测'] * df_enhanced['renewable_penetration']
        df_enhanced['weekend_peak_interaction'] = df_enhanced['is_weekend'] * df_enhanced['is_peak']
        df_enhanced['price_level_volatility'] = df_enhanced['rt_price_ma_24'] * df_enhanced['price_volatility_12h']
        
        # Remove rows with NaN (due to lags and rolling windows)
        df_enhanced = df_enhanced.dropna()
        
        print(f"Enhanced dataset shape: {df_enhanced.shape}")
        print(f"Added {df_enhanced.shape[1] - df.shape[1]} new features")
        
        return df_enhanced
    
    def select_best_features(self, df, target_col='实时出清电价', top_k=30):
        """Select most predictive features"""
        # Calculate correlations
        correlations = df.corr()[target_col].abs().sort_values(ascending=False)
        
        # Select top features (excluding target itself)
        best_features = correlations[1:top_k+1].index.tolist()
        
        print(f"Selected top {len(best_features)} features:")
        for i, feature in enumerate(best_features[:10], 1):
            print(f"  {i:2d}. {feature} (corr: {correlations[feature]:.3f})")
        
        return best_features
    
    def custom_mape_loss(self, y_true, y_pred):
        """Custom loss function that directly optimizes for MAPE"""
        # Add small epsilon to avoid division by zero
        epsilon = 1e-7
        diff = tf.abs(y_true - y_pred)
        percentage_error = diff / (tf.abs(y_true) + epsilon)
        return tf.reduce_mean(percentage_error)
    
    def custom_weighted_loss(self, y_true, y_pred):
        """Weighted loss that penalizes low-price prediction errors less"""
        # Convert scaled values back to original scale for weighting
        mse = tf.square(y_true - y_pred)
        
        # Weight based on price level (less penalty for low prices)
        weights = tf.maximum(tf.abs(y_true), 0.1)  # Minimum weight of 0.1
        weighted_mse = mse / weights
        
        return tf.reduce_mean(weighted_mse)
    
    def build_improved_cnn(self, input_shape, learning_rate=0.001, dropout=0.3):
        """Enhanced CNN with deeper architecture and regularization"""
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            
            # First CNN block
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Dropout(dropout),
            
            # Second CNN block
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Dropout(dropout),
            
            # Third CNN block
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            layers.Dense(64, activation='relu'),
            layers.Dropout(dropout/2),
            layers.Dense(self.prediction_horizon)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self.custom_mape_loss,
            metrics=['mae', 'mse']
        )
        
        return model
    
    def build_attention_cnn(self, input_shape, learning_rate=0.001):
        """CNN with attention mechanism"""
        inputs = layers.Input(shape=input_shape)
        
        # CNN feature extraction
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2)(x)
        
        # Attention mechanism
        attention = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
        x = layers.Add()([x, attention])
        x = layers.LayerNormalization()(x)
        
        # Final processing
        x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.prediction_horizon)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self.custom_mape_loss,
            metrics=['mae', 'mse']
        )
        
        return model
    
    def build_ensemble_model(self, input_shape, learning_rate=0.001):
        """Ensemble of CNN and LSTM"""
        inputs = layers.Input(shape=input_shape)
        
        # CNN branch
        cnn_x = layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
        cnn_x = layers.MaxPooling1D(2)(cnn_x)
        cnn_x = layers.Conv1D(32, 3, activation='relu', padding='same')(cnn_x)
        cnn_x = layers.GlobalAveragePooling1D()(cnn_x)
        
        # LSTM branch
        lstm_x = layers.LSTM(32, return_sequences=False)(inputs)
        
        # Combine branches
        combined = layers.Concatenate()([cnn_x, lstm_x])
        x = layers.Dense(64, activation='relu')(combined)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(self.prediction_horizon)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self.custom_mape_loss,
            metrics=['mae', 'mse']
        )
        
        return model
    
    def prepare_improved_data(self, data_path='cleaned_data/energy_data_cleaned.csv'):
        """Prepare data with advanced features"""
        print("Loading and preparing enhanced dataset...")
        
        # Load data
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Create advanced features
        df_enhanced = self.advanced_feature_engineering(df)
        
        # Select best features
        target_col = '实时出清电价'
        feature_cols = self.select_best_features(df_enhanced, target_col, top_k=25)
        
        # Prepare arrays
        self.X = df_enhanced[feature_cols].values
        self.y = df_enhanced[target_col].values.reshape(-1, 1)
        
        # Scale data
        self.X_scaled = self.scaler_features.fit_transform(self.X)
        self.y_scaled = self.scaler_target.fit_transform(self.y)
        
        print(f"Final feature set: {len(feature_cols)} features")
        print(f"Data shape: X={self.X_scaled.shape}, y={self.y_scaled.shape}")
        
        return self.X_scaled, self.y_scaled.flatten(), feature_cols
    
    def create_sequences(self, X, y):
        """Create sequences for time series prediction"""
        X_seq, y_seq = [], []
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    def train_test_split(self, X, y, train_size=0.7, val_size=0.15):
        """Time-based splits"""
        # Create sequences
        X_seq, y_seq = self.create_sequences(X, y)
        
        # Split
        n = len(X_seq)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))
        
        return (X_seq[:train_end], y_seq[:train_end],
                X_seq[train_end:val_end], y_seq[train_end:val_end],
                X_seq[val_end:], y_seq[val_end:])
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Enhanced evaluation with focus on MAPE"""
        # Predict
        y_pred_scaled = model.predict(X_test, verbose=0)
        
        # Inverse transform
        y_pred = self.scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = self.scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE with special handling for low prices
        mape_all = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # MAPE for different price ranges
        low_price_mask = y_true < 50
        high_price_mask = y_true >= 50
        
        if np.sum(low_price_mask) > 0:
            mape_low = np.mean(np.abs((y_true[low_price_mask] - y_pred[low_price_mask]) / y_true[low_price_mask])) * 100
        else:
            mape_low = 0
            
        if np.sum(high_price_mask) > 0:
            mape_high = np.mean(np.abs((y_true[high_price_mask] - y_pred[high_price_mask]) / y_true[high_price_mask])) * 100
        else:
            mape_high = 0
        
        metrics = {
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE_All': mape_all,
            'MAPE_Low_Price': mape_low,
            'MAPE_High_Price': mape_high,
            'Low_Price_Periods': np.sum(low_price_mask)
        }
        
        return metrics, y_true, y_pred
    
    def run_improved_experiment(self):
        """Run comprehensive improvement experiment"""
        print("=== IMPROVED NEURAL NETWORK EXPERIMENT ===\n")
        
        # Prepare enhanced data
        X, y, feature_cols = self.prepare_improved_data()
        
        # Create splits
        X_train, y_train, X_val, y_val, X_test, y_test = self.train_test_split(X, y)
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Model configurations to test
        models_config = {
            'Improved_CNN': self.build_improved_cnn,
            'Attention_CNN': self.build_attention_cnn,
            'Ensemble_CNN_LSTM': self.build_ensemble_model
        }
        
        results = []
        
        # Train models
        for model_name, model_builder in models_config.items():
            print(f"\n--- Training {model_name} ---")
            
            # Build model
            model = model_builder(input_shape)
            
            # Training callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
                )
            ]
            
            # Train
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate
            metrics, y_true, y_pred = self.evaluate_model(model, X_test, y_test, model_name)
            results.append(metrics)
            
            print(f"Results: MAPE={metrics['MAPE_All']:.2f}%, RMSE={metrics['RMSE']:.2f}, R²={metrics['R²']:.3f}")
            
            # Store for comparison
            self.models[model_name] = model
            self.results[model_name] = {'metrics': metrics, 'predictions': (y_true, y_pred)}
        
        # Results comparison
        results_df = pd.DataFrame(results)
        print(f"\n{'='*80}")
        print("IMPROVED MODELS COMPARISON")
        print(f"{'='*80}")
        print(results_df.round(3))
        
        # Best model identification
        best_mape = results_df.loc[results_df['MAPE_All'].idxmin()]
        print(f"\n🏆 BEST MAPE: {best_mape['Model']} ({best_mape['MAPE_All']:.2f}%)")
        print(f"   RMSE: {best_mape['RMSE']:.2f}, R²: {best_mape['R²']:.3f}")
        
        # Save results
        results_df.to_csv('improved_model_results.csv', index=False)
        
        return results_df

def main():
    """Run improved model experiment"""
    predictor = ImprovedEnergyPredictor(sequence_length=24)
    results = predictor.run_improved_experiment()
    return predictor, results

if __name__ == "__main__":
    predictor, results = main()