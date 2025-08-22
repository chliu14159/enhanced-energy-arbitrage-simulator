import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load the processed data
data_path = "/Users/randomwalk/Documents/CODE/REPO/energy_trading_js/processed/wind_solar/wind_solar_data_cleaned_20250822_200203.parquet"
df = pd.read_parquet(data_path)

print("🌪️ WIND AND SOLAR DATA ANALYSIS")
print("=" * 50)

# Basic info
print(f"\n📊 Dataset Overview:")
print(f"Total records: {len(df):,}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"Stations: {df['station_id'].nunique()}")
print(f"Time span: {(df['datetime'].max() - df['datetime'].min()).days} days")

# Station mapping based on the data patterns
station_info = {
    '501974': {'name': 'Wind Station A', 'type': 'wind', 'capacity_estimate': '~12 MW'},
    '502633': {'name': 'Solar Station A', 'type': 'solar', 'capacity_estimate': '~32 MW'},
    '505519': {'name': 'Solar Station B', 'type': 'solar', 'capacity_estimate': '~14 MW'},
    '506445': {'name': 'Wind Station B', 'type': 'wind', 'capacity_estimate': '~20 MW'}
}

print(f"\n🏭 Station Information:")
for station_id, info in station_info.items():
    station_data = df[df['station_id'] == station_id]
    print(f"  {station_id} ({info['name']}):")
    print(f"    Type: {info['type'].title()}")
    print(f"    Records: {len(station_data):,}")
    print(f"    Max Output: {station_data['value'].max():.1f} MW")
    print(f"    Avg Output: {station_data['value'].mean():.1f} MW")
    print(f"    Capacity Factor: {(station_data['value'].mean() / station_data['value'].max() * 100):.1f}%")

# Analyze daily patterns
print(f"\n⏰ Daily Generation Patterns:")
hourly_avg = df.groupby(['station_id', 'hour'])['value'].mean().unstack(level=0)

# Solar patterns (should peak around noon)
solar_stations = ['502633', '505519']
wind_stations = ['501974', '506445']

print(f"\n☀️ Solar Peak Hours:")
for station in solar_stations:
    if station in hourly_avg.columns:
        peak_hour = hourly_avg[station].idxmax()
        peak_value = hourly_avg[station].max()
        print(f"  Station {station}: Peak at {peak_hour}:00 with {peak_value:.0f} MW")

print(f"\n🌪️ Wind Patterns:")
for station in wind_stations:
    if station in hourly_avg.columns:
        avg_value = hourly_avg[station].mean()
        std_value = hourly_avg[station].std()
        print(f"  Station {station}: Avg {avg_value:.0f} MW, Variability ±{std_value:.0f} MW")

# Check for data quality issues
print(f"\n🔍 Data Quality Analysis:")
print(f"\nMissing Values:")
for station in df['station_id'].unique():
    station_data = df[df['station_id'] == station]
    missing_hours = 0
    # Calculate expected vs actual records
    date_range = (station_data['datetime'].max() - station_data['datetime'].min()).total_seconds() / 3600
    expected_records = int(date_range)
    actual_records = len(station_data)
    missing_pct = (1 - actual_records / expected_records) * 100 if expected_records > 0 else 0
    print(f"  Station {station}: {missing_pct:.1f}% missing data")

# Generate some insights for modeling
print(f"\n💡 Key Insights for Modeling:")

# Solar capacity factors
solar_data = df[df['type'] == 'solar']
solar_avg_cf = solar_data.groupby('station_id')['value'].mean()
solar_max_cap = solar_data.groupby('station_id')['value'].max()
solar_cf = (solar_avg_cf / solar_max_cap * 100)

print(f"\n☀️ Solar Capacity Factors:")
for station, cf in solar_cf.items():
    print(f"  Station {station}: {cf:.1f}%")

# Wind capacity factors  
wind_data = df[df['type'] == 'wind']
wind_avg_cf = wind_data.groupby('station_id')['value'].mean()
wind_max_cap = wind_data.groupby('station_id')['value'].max()
wind_cf = (wind_avg_cf / wind_max_cap * 100)

print(f"\n🌪️ Wind Capacity Factors:")
for station, cf in wind_cf.items():
    print(f"  Station {station}: {cf:.1f}%")

# Correlation between stations
print(f"\n🔗 Inter-station Correlations:")
pivot_data = df.pivot_table(index='datetime', columns='station_id', values='value', aggfunc='mean')
correlations = pivot_data.corr()

print("Station correlation matrix:")
print(correlations.round(3))

# Seasonal patterns (if enough data)
df['month'] = df['datetime'].dt.month
monthly_avg = df.groupby(['station_id', 'month'])['value'].mean().unstack(level=0)

print(f"\n📅 Monthly Averages (MW):")
if len(monthly_avg) > 1:
    print(monthly_avg.round(0))
else:
    print("Not enough data for seasonal analysis")

print(f"\n✅ Data Processing Complete!")
print(f"💾 Cleaned dataset ready for renewable energy forecasting models")
print(f"📈 Next steps: Use this data to enhance the arbitrage model with renewable forecasting")
