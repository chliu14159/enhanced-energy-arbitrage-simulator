import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def analyze_prediction_errors():
    """Analyze where the high MAPE is coming from"""
    print("=== MAPE ANALYSIS: Understanding High Error Rates ===\n")
    
    # Load the original data
    df = pd.read_csv('cleaned_data/energy_data_cleaned.csv', index_col=0, parse_dates=True)
    
    # Analyze price characteristics that might cause high MAPE
    target = '实时出清电价'
    prices = df[target]
    
    print("1. PRICE DISTRIBUTION ANALYSIS")
    print(f"Mean price: {prices.mean():.2f} RMB/MWh")
    print(f"Median price: {prices.median():.2f} RMB/MWh")
    print(f"Std deviation: {prices.std():.2f} RMB/MWh")
    print(f"Min price: {prices.min():.2f} RMB/MWh")
    print(f"Max price: {prices.max():.2f} RMB/MWh")
    print(f"Price range: {prices.max() - prices.min():.2f} RMB/MWh")
    
    # Identify low price periods (main MAPE driver)
    low_price_threshold = 50  # RMB/MWh
    low_prices = prices[prices < low_price_threshold]
    print(f"\n2. LOW PRICE ANALYSIS (< {low_price_threshold} RMB/MWh)")
    print(f"Low price periods: {len(low_prices)} / {len(prices)} ({len(low_prices)/len(prices)*100:.1f}%)")
    print(f"Avg low price: {low_prices.mean():.2f} RMB/MWh")
    print(f"Min price: {low_prices.min():.2f} RMB/MWh")
    
    # Price volatility analysis
    price_changes = prices.pct_change().dropna()
    print(f"\n3. VOLATILITY ANALYSIS")
    print(f"Avg price change: {price_changes.mean()*100:.2f}%")
    print(f"Price change std: {price_changes.std()*100:.2f}%")
    print(f"Max price increase: {price_changes.max()*100:.1f}%")
    print(f"Max price decrease: {price_changes.min()*100:.1f}%")
    
    # Extreme events
    extreme_changes = price_changes[abs(price_changes) > 0.5]  # 50%+ changes
    print(f"Extreme changes (>50%): {len(extreme_changes)} events")
    
    # Time-based patterns that affect MAPE
    print(f"\n4. TEMPORAL PATTERNS")
    hourly_stats = df.groupby('hour')[target].agg(['mean', 'std', 'min', 'max'])
    most_volatile_hour = hourly_stats['std'].idxmax()
    least_volatile_hour = hourly_stats['std'].idxmin()
    print(f"Most volatile hour: {most_volatile_hour}:00 (σ={hourly_stats.loc[most_volatile_hour, 'std']:.2f})")
    print(f"Least volatile hour: {least_volatile_hour}:00 (σ={hourly_stats.loc[least_volatile_hour, 'std']:.2f})")
    
    # Weekend vs weekday volatility
    weekend_volatility = df[df['is_weekend'] == 1][target].std()
    weekday_volatility = df[df['is_weekend'] == 0][target].std()
    print(f"Weekend volatility: {weekend_volatility:.2f}")
    print(f"Weekday volatility: {weekday_volatility:.2f}")
    
    return df, prices, low_prices

def identify_improvement_opportunities():
    """Identify specific areas for model improvement"""
    print(f"\n{'='*60}")
    print("IMPROVEMENT OPPORTUNITIES ANALYSIS")
    print(f"{'='*60}")
    
    df, prices, low_prices = analyze_prediction_errors()
    
    print("\n🎯 MAIN MAPE DRIVERS IDENTIFIED:")
    print("1. LOW PRICE PERIODS - Small absolute errors become large % errors")
    print("   - Solution: Custom loss function, separate models for different price ranges")
    
    print("2. HIGH VOLATILITY - Extreme price swings are hard to predict")
    print("   - Solution: Volatility modeling, ensemble methods, longer sequences")
    
    print("3. LIMITED FEATURES - Missing key market information")
    print("   - Solution: Add lag features, weather data, market indicators")
    
    print("4. SIMPLE ARCHITECTURE - Current models may be too basic")
    print("   - Solution: Deeper networks, attention mechanisms, residual connections")
    
    print("\n📊 SPECIFIC RECOMMENDATIONS:")
    
    # Calculate potential improvements
    low_price_periods = len(low_prices) / len(prices) * 100
    print(f"1. Handle {low_price_periods:.1f}% of low-price periods differently")
    print("   - Use separate model or weighted loss for prices < 50 RMB/MWh")
    
    volatility_reduction_potential = df['实时出清电价'].std() * 0.3  # 30% reduction target
    print(f"2. Target volatility reduction: {volatility_reduction_potential:.2f} RMB/MWh")
    print("   - Better features could reduce unexplained variance")
    
    print("3. Feature engineering priorities:")
    corr_with_price = df.corr()['实时出清电价'].abs().sort_values(ascending=False)
    print("   - Top correlations:", corr_with_price.head(5).to_dict())
    
    return df

def calculate_mape_improvement_potential():
    """Calculate realistic MAPE improvement targets"""
    df = pd.read_csv('cleaned_data/energy_data_cleaned.csv', index_col=0, parse_dates=True)
    prices = df['实时出清电价']
    
    print(f"\n{'='*50}")
    print("MAPE IMPROVEMENT POTENTIAL")
    print(f"{'='*50}")
    
    # Current baseline: Day-ahead price as predictor
    da_prices = df['日前出清电价']
    baseline_mape = np.mean(np.abs((prices - da_prices) / prices)) * 100
    print(f"Baseline MAPE (using DA price): {baseline_mape:.2f}%")
    
    # Best possible with perfect hourly patterns
    hourly_medians = df.groupby('hour')['实时出清电价'].median()
    hourly_predictions = df['hour'].map(hourly_medians)
    hourly_mape = np.mean(np.abs((prices - hourly_predictions) / prices)) * 100
    print(f"Hourly pattern MAPE: {hourly_mape:.2f}%")
    
    # Target improvement
    current_best_mape = 20.89  # From Transformer
    target_mape = 15.0  # Ambitious but realistic target
    improvement_needed = current_best_mape - target_mape
    
    print(f"\nCurrent best MAPE: {current_best_mape:.2f}%")
    print(f"Target MAPE: {target_mape:.2f}%")
    print(f"Improvement needed: {improvement_needed:.2f} percentage points")
    print(f"Relative improvement: {improvement_needed/current_best_mape*100:.1f}%")
    
    return target_mape

if __name__ == "__main__":
    df = identify_improvement_opportunities()
    target_mape = calculate_mape_improvement_potential()