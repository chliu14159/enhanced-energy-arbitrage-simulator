import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

def compare_all_results():
    """Compare original vs improved models"""
    print("=== COMPREHENSIVE MODEL COMPARISON ===\n")
    
    # Load original results
    original_results = pd.read_csv('model_comparison_summary.csv')
    print("ORIGINAL MODELS (Simple Features):")
    print(original_results[['Model', 'MAE', 'RMSE', 'R²', 'MAPE']].round(3))
    
    # Load improved results
    improved_results = pd.read_csv('improved_model_results.csv')
    print("\nIMPROVED MODELS (Advanced Features + Custom Loss):")
    print(improved_results[['Model', 'MAE', 'RMSE', 'R²', 'MAPE_All']].round(3))
    
    # Analysis
    print("\n=== ANALYSIS OF RESULTS ===")
    
    best_original_mape = original_results['MAPE'].min()
    best_improved_mape = improved_results['MAPE_All'].min()
    
    print(f"Best Original MAPE: {best_original_mape:.2f}%")
    print(f"Best Improved MAPE: {best_improved_mape:.2f}%")
    
    if best_improved_mape > best_original_mape:
        print("❌ IMPROVED MODELS PERFORMED WORSE!")
        print("\n🔍 POTENTIAL CAUSES:")
        print("1. OVERFITTING - Too many features (25 vs 7 original)")
        print("2. CUSTOM LOSS - MAPE loss may be unstable during training")
        print("3. FEATURE NOISE - Lag features may introduce noise")
        print("4. MODEL COMPLEXITY - Deeper networks may overfit small dataset")
        
        print("\n💡 RECOMMENDATIONS:")
        print("1. Try simpler models with fewer, carefully selected features")
        print("2. Use original MSE loss with MAPE evaluation")
        print("3. Focus on feature selection rather than feature creation")
        print("4. Try traditional ML methods (Random Forest, XGBoost)")
        
    return original_results, improved_results

def optimized_simple_model():
    """Create optimized model with careful feature selection"""
    print("\n=== OPTIMIZED SIMPLE MODEL ===")
    
    # Load data
    df = pd.read_csv('cleaned_data/energy_data_cleaned.csv', index_col=0, parse_dates=True)
    
    # Carefully selected features based on EDA insights
    features = [
        '日前出清电价',      # Strongest predictor (r=0.879)
        '新能源预测',        # Strong negative correlation  
        '竞价空间(火电)',    # Strong positive correlation
        '负荷预测',          # Market fundamental
        'hour',              # Time pattern
        'is_peak',           # Peak/off-peak
        'is_weekend'         # Weekend effect
    ]
    
    target = '实时出清电价'
    
    # Add simple lag features (just 1 and 4 periods)
    df['rt_price_lag1'] = df[target].shift(1)
    df['da_price_lag1'] = df['日前出清电价'].shift(1)
    df['rt_price_lag4'] = df[target].shift(4)  # 1 hour ago
    
    # Add these to features
    features.extend(['rt_price_lag1', 'da_price_lag1', 'rt_price_lag4'])
    
    # Remove NaN
    df = df.dropna()
    
    X = df[features].values
    y = df[target].values
    
    # Scale data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    print(f"Using {len(features)} carefully selected features")
    print("Features:", features)
    
    return X_scaled, y_scaled, scaler_X, scaler_y, features

def create_sequences_simple(X, y, seq_length=12):
    """Create sequences with shorter length"""
    X_seq, y_seq = [], []
    for i in range(seq_length, len(X)):
        X_seq.append(X[i-seq_length:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

def build_optimized_cnn(input_shape, learning_rate=0.001):
    """Simpler CNN with standard MSE loss"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # Simpler architecture
        layers.Conv1D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, 3, activation='relu', padding='same'),
        layers.GlobalAveragePooling1D(),
        
        # Simple dense layers
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',  # Back to MSE loss
        metrics=['mae']
    )
    
    return model

def build_traditional_ml_models(X_train, y_train, X_test, y_test, scaler_y):
    """Try traditional ML models"""
    print("\n=== TRADITIONAL ML MODELS ===")
    
    results = []
    
    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    
    # Convert back to original scale
    y_true_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = scaler_y.inverse_transform(y_pred_rf.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mae_rf = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse_rf = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2_rf = r2_score(y_true_orig, y_pred_orig)
    mape_rf = np.mean(np.abs((y_true_orig - y_pred_orig) / y_true_orig)) * 100
    
    results.append({
        'Model': 'Random_Forest',
        'MAE': mae_rf,
        'RMSE': rmse_rf,
        'R²': r2_rf,
        'MAPE': mape_rf
    })
    
    print(f"Random Forest - MAPE: {mape_rf:.2f}%, RMSE: {rmse_rf:.2f}, R²: {r2_rf:.3f}")
    
    # Ridge Regression
    print("Training Ridge Regression...")
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    
    y_pred_ridge = ridge.predict(X_test)
    y_pred_ridge_orig = scaler_y.inverse_transform(y_pred_ridge.reshape(-1, 1)).flatten()
    
    mae_ridge = mean_absolute_error(y_true_orig, y_pred_ridge_orig)
    rmse_ridge = np.sqrt(mean_squared_error(y_true_orig, y_pred_ridge_orig))
    r2_ridge = r2_score(y_true_orig, y_pred_ridge_orig)
    mape_ridge = np.mean(np.abs((y_true_orig - y_pred_ridge_orig) / y_true_orig)) * 100
    
    results.append({
        'Model': 'Ridge_Regression',
        'MAE': mae_ridge,
        'RMSE': rmse_ridge,
        'R²': r2_ridge,
        'MAPE': mape_ridge
    })
    
    print(f"Ridge Regression - MAPE: {mape_ridge:.2f}%, RMSE: {rmse_ridge:.2f}, R²: {r2_ridge:.3f}")
    
    return results

def final_optimization_experiment():
    """Run final optimization experiment"""
    print("=== FINAL MAPE OPTIMIZATION EXPERIMENT ===\n")
    
    # First compare all previous results
    original_results, improved_results = compare_all_results()
    
    # Prepare optimized data
    X, y, scaler_X, scaler_y, features = optimized_simple_model()
    
    # Create sequences
    X_seq, y_seq = create_sequences_simple(X, y, seq_length=12)
    
    # Split data
    n = len(X_seq)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    X_train, y_train = X_seq[:train_end], y_seq[:train_end]
    X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
    X_test, y_test = X_seq[val_end:], y_seq[val_end:]
    
    print(f"\nData splits: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    # Try traditional ML on non-sequence data
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    ml_results = build_traditional_ml_models(X_train_flat, y_train, X_test_flat, y_test, scaler_y)
    
    # Try optimized CNN
    print("\n=== OPTIMIZED CNN ===")
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    model = build_optimized_cnn(input_shape)
    
    # Train with early stopping
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )
    
    # Evaluate CNN
    y_pred_cnn = model.predict(X_test, verbose=0)
    y_true_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = scaler_y.inverse_transform(y_pred_cnn.reshape(-1, 1)).flatten()
    
    mae_cnn = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse_cnn = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2_cnn = r2_score(y_true_orig, y_pred_orig)
    mape_cnn = np.mean(np.abs((y_true_orig - y_pred_orig) / y_true_orig)) * 100
    
    ml_results.append({
        'Model': 'Optimized_CNN',
        'MAE': mae_cnn,
        'RMSE': rmse_cnn,
        'R²': r2_cnn,
        'MAPE': mape_cnn
    })
    
    print(f"Optimized CNN - MAPE: {mape_cnn:.2f}%, RMSE: {rmse_cnn:.2f}, R²: {r2_cnn:.3f}")
    
    # Final comparison
    print(f"\n{'='*60}")
    print("FINAL OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    
    final_results = pd.DataFrame(ml_results)
    print(final_results.round(3))
    
    # Best model
    best_mape_idx = final_results['MAPE'].idxmin()
    best_model = final_results.iloc[best_mape_idx]
    
    print(f"\n🏆 BEST MAPE MODEL: {best_model['Model']}")
    print(f"   MAPE: {best_model['MAPE']:.2f}%")
    print(f"   RMSE: {best_model['RMSE']:.2f}")
    print(f"   R²: {best_model['R²']:.3f}")
    
    # Improvement analysis
    original_best_mape = original_results['MAPE'].min()
    final_best_mape = final_results['MAPE'].min()
    
    improvement = original_best_mape - final_best_mape
    improvement_pct = improvement / original_best_mape * 100
    
    print(f"\n📈 IMPROVEMENT SUMMARY:")
    print(f"Original best MAPE: {original_best_mape:.2f}%")
    print(f"Final best MAPE: {final_best_mape:.2f}%")
    if improvement > 0:
        print(f"Improvement: {improvement:.2f} percentage points ({improvement_pct:.1f}% relative)")
    else:
        print(f"No improvement achieved. Consider other approaches.")
    
    # Save results
    final_results.to_csv('final_optimization_results.csv', index=False)
    
    # Practical recommendations
    print(f"\n💡 PRACTICAL RECOMMENDATIONS:")
    
    if final_best_mape < 18:
        print("✅ EXCELLENT: MAPE < 18% - Ready for production")
    elif final_best_mape < 22:
        print("✅ GOOD: MAPE < 22% - Acceptable for most use cases")
    elif final_best_mape < 25:
        print("⚠️ FAIR: MAPE < 25% - Use with caution, consider ensemble")
    else:
        print("❌ POOR: MAPE > 25% - Need different approach")
    
    print("\n🚀 NEXT STEPS:")
    if best_model['Model'] == 'Random_Forest':
        print("- Random Forest works best - consider XGBoost or LightGBM")
        print("- Feature importance analysis for better understanding")
        print("- Hyperparameter tuning for Random Forest")
    elif best_model['Model'] == 'Ridge_Regression':
        print("- Linear model works well - simple and interpretable")
        print("- Consider polynomial features or interactions")
        print("- Try ElasticNet for feature selection")
    else:
        print("- Neural network approach is promising")
        print("- Try different architectures and hyperparameters")
        print("- Consider ensemble of neural networks")
    
    return final_results

if __name__ == "__main__":
    results = final_optimization_experiment()