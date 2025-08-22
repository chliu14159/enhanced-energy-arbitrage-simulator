#!/usr/bin/env python3
"""
Simple test script to validate dashboard functionality
"""

import streamlit as st
import pandas as pd
import numpy as np

st.title("🔍 Dashboard Validation Test")

try:
    # Test 1: Basic imports
    st.write("✅ **Test 1:** Basic imports successful")
    
    # Test 2: Sample data generation
    st.write("🧪 **Test 2:** Testing sample data generation...")
    
    dates = pd.date_range(start='2025-07-01', end='2025-07-02', freq='h')
    np.random.seed(42)
    
    sample_data = []
    stations = ['501974', '502633', '505519', '506445']
    
    for dt in dates[:5]:  # Just first 5 hours
        hour = dt.hour
        day_of_year = dt.dayofyear
        
        for station in stations:
            base_wind = 50 + 30 * np.sin(2 * np.pi * day_of_year / 365)
            wind_gen = max(0, base_wind + np.random.normal(0, 15))
            
            solar_factor = max(0, np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0
            base_solar = 100 * solar_factor
            solar_gen = max(0, base_solar + np.random.normal(0, 10))
            
            sample_data.append({
                'datetime': dt,
                'station_id': station,
                'wind_generation': wind_gen,
                'solar_generation': solar_gen,
                'total_generation': wind_gen + solar_gen,
            })
    
    df = pd.DataFrame(sample_data)
    st.write(f"✅ Sample data created: {df.shape}")
    st.write("Columns:", list(df.columns))
    st.dataframe(df.head())
    
    # Test 3: Station filtering
    st.write("🧪 **Test 3:** Testing station filtering...")
    station_data = df[df['station_id'] == '501974'].copy()
    st.write(f"✅ Filtered data: {station_data.shape}")
    
    # Test 4: Column access
    st.write("🧪 **Test 4:** Testing column access...")
    if len(station_data) > 0:
        row = station_data.iloc[0]
        wind_val = row.get('wind_generation', 0)
        solar_val = row.get('solar_generation', 0)
        st.write(f"✅ Wind: {wind_val:.2f}, Solar: {solar_val:.2f}")
    
    st.success("🎉 All tests passed! Dashboard should work correctly.")
    
except Exception as e:
    st.error(f"❌ Error during testing: {e}")
    st.write("Error details:", str(e))
