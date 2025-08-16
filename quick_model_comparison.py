import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

def load_and_prepare_data():
    """Load and prepare energy data for modeling"""
    print("Loading data...")
    df = pd.read_csv('cleaned_data/energy_data_cleaned.csv', index_col=0, parse_dates=True)
    
    # Key features based on EDA
    features = ['日前出清电价', '新能源预测', '竞价空间(火电)', '负荷预测', 
               'hour', 'is_peak', 'is_weekend']
    target = '实时出清电价'
    
    X = df[features].values
    y = df[target].values
    
    # Scale data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X_scaled, y_scaled, scaler_X, scaler_y, features

def create_sequences(X, y, seq_length=12):
    """Create sequences for time series models"""
    X_seq, y_seq = [], []
    for i in range(seq_length, len(X)):
        X_seq.append(X[i-seq_length:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

def split_data(X, y, train_size=0.7, val_size=0.15):
    """Time-based train/val/test split"""
    n = len(X)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    
    return (X[:train_end], y[:train_end],
            X[train_end:val_end], y[train_end:val_end],
            X[val_end:], y[val_end:])

def build_lstm_model(input_shape):
    """Simple LSTM model"""
    model = keras.Sequential([
        layers.LSTM(32, input_shape=input_shape),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_cnn_model(input_shape):
    """Simple CNN model"""
    model = keras.Sequential([
        layers.Conv1D(32, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Conv1D(16, 3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_gru_model(input_shape):
    """Simple GRU model"""
    model = keras.Sequential([
        layers.GRU(32, input_shape=input_shape),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_transformer_model(input_shape):
    """Simple Transformer model"""
    inputs = layers.Input(shape=input_shape)
    
    # Multi-head attention
    attention = layers.MultiHeadAttention(num_heads=2, key_dim=16)(inputs, inputs)
    attention = layers.LayerNormalization()(attention + inputs)
    
    # Feed forward
    ff = layers.Dense(32, activation='relu')(attention)
    ff = layers.Dense(input_shape[-1])(ff)
    outputs = layers.LayerNormalization()(ff + attention)
    
    # Global pooling and final layers
    pooled = layers.GlobalAveragePooling1D()(outputs)
    final = layers.Dense(16, activation='relu')(pooled)
    predictions = layers.Dense(1)(final)
    
    model = keras.Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def evaluate_model(model, X_test, y_test, scaler_y, model_name):
    """Evaluate model and return metrics"""
    # Predict
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Inverse transform
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape
    }, y_true, y_pred

def main():
    """Run quick model comparison"""
    print("=== ENERGY PRICE PREDICTION MODEL COMPARISON ===\n")
    
    # Load data
    X, y, scaler_X, scaler_y, features = load_and_prepare_data()
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Create sequences
    X_seq, y_seq = create_sequences(X, y, seq_length=12)
    print(f"Sequence shape: X={X_seq.shape}, y={y_seq.shape}")
    
    # Split data
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X_seq, y_seq)
    print(f"Splits - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Model configurations
    input_shape = (X_train.shape[1], X_train.shape[2])
    models_config = {
        'LSTM': build_lstm_model,
        'CNN': build_cnn_model,
        'GRU': build_gru_model,
        'Transformer': build_transformer_model
    }
    
    results = []
    predictions = {}
    
    # Train and evaluate each model
    for model_name, model_builder in models_config.items():
        print(f"\n--- Training {model_name} ---")
        start_time = time.time()
        
        # Build model
        model = model_builder(input_shape)
        print(f"Model parameters: {model.count_params():,}")
        
        # Train with early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        training_time = time.time() - start_time
        
        # Evaluate
        metrics, y_true, y_pred = evaluate_model(model, X_test, y_test, scaler_y, model_name)
        metrics['Training_Time'] = training_time
        metrics['Epochs'] = len(history.history['loss'])
        
        results.append(metrics)
        predictions[model_name] = {'y_true': y_true, 'y_pred': y_pred}
        
        print(f"Completed in {training_time:.1f}s ({metrics['Epochs']} epochs)")
        print(f"MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, R²: {metrics['R²']:.3f}")
    
    # Results summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS COMPARISON")
    print(f"{'='*60}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.round(3)
    print(results_df.to_string(index=False))
    
    # Best models
    print(f"\n{'='*30}")
    print("BEST MODELS BY METRIC:")
    print(f"{'='*30}")
    
    best_mae = results_df.loc[results_df['MAE'].idxmin()]
    best_rmse = results_df.loc[results_df['RMSE'].idxmin()]
    best_r2 = results_df.loc[results_df['R²'].idxmax()]
    best_mape = results_df.loc[results_df['MAPE'].idxmin()]
    
    print(f"Best MAE: {best_mae['Model']} ({best_mae['MAE']:.2f} RMB/MWh)")
    print(f"Best RMSE: {best_rmse['Model']} ({best_rmse['RMSE']:.2f} RMB/MWh)")
    print(f"Best R²: {best_r2['Model']} ({best_r2['R²']:.3f})")
    print(f"Best MAPE: {best_mape['Model']} ({best_mape['MAPE']:.2f}%)")
    
    # Quick visualization
    plt.figure(figsize=(15, 10))
    
    # Metrics comparison
    metrics_to_plot = ['MAE', 'RMSE', 'R²', 'MAPE']
    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(2, 3, i)
        bars = plt.bar(results_df['Model'], results_df[metric])
        plt.title(f'{metric} Comparison')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        
        # Highlight best
        if metric == 'R²':
            best_idx = results_df[metric].idxmax()
        else:
            best_idx = results_df[metric].idxmin()
        bars[best_idx].set_color('green')
    
    # Training time comparison
    plt.subplot(2, 3, 5)
    plt.bar(results_df['Model'], results_df['Training_Time'])
    plt.title('Training Time (seconds)')
    plt.ylabel('Time (s)')
    plt.xticks(rotation=45)
    
    # Sample predictions
    plt.subplot(2, 3, 6)
    best_model = best_rmse['Model']
    pred_data = predictions[best_model]
    sample_size = min(100, len(pred_data['y_true']))
    plt.plot(pred_data['y_true'][:sample_size], label='Actual', alpha=0.8)
    plt.plot(pred_data['y_pred'][:sample_size], label='Predicted', alpha=0.8)
    plt.title(f'{best_model} - Sample Predictions')
    plt.xlabel('Time Steps')
    plt.ylabel('Price (RMB/MWh)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('quick_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    results_df.to_csv('model_comparison_summary.csv', index=False)
    print(f"\nResults saved to 'model_comparison_summary.csv'")
    
    return results_df, predictions

if __name__ == "__main__":
    results, predictions = main()