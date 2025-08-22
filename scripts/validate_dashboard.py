#!/usr/bin/env python3
"""
Dashboard Validation Script
==========================

Quick validation to ensure all components for the renewable forecasting dashboard are working.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def validate_dashboard_requirements():
    """Validate that all requirements for the dashboard are met."""
    print("🔍 Validating Renewable Forecasting Dashboard Requirements...")
    
    issues = []
    
    # Check data file
    data_path = "/Users/randomwalk/Documents/CODE/REPO/energy_trading_js/processed/wind_solar/wind_solar_data_cleaned_20250822_200203.parquet"
    if Path(data_path).exists():
        print("✅ Data file found")
        try:
            df = pd.read_parquet(data_path)
            print(f"✅ Data loaded successfully: {len(df):,} records")
            print(f"✅ Stations available: {df['station_id'].unique()}")
            print(f"✅ Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        except Exception as e:
            issues.append(f"❌ Error loading data: {e}")
    else:
        issues.append("❌ Data file not found")
    
    # Check models directory
    models_dir = "/Users/randomwalk/Documents/CODE/REPO/energy_trading_js/models/lstm_forecasting"
    if Path(models_dir).exists():
        print("✅ Models directory found")
        
        # Check for metadata file
        metadata_path = Path(models_dir) / "model_metadata.pkl"
        if metadata_path.exists():
            print("✅ Model metadata found")
        else:
            issues.append("⚠️ Model metadata not found (dashboard will work with limited functionality)")
    else:
        issues.append("⚠️ Models directory not found (dashboard will work with limited functionality)")
    
    # Check required packages
    required_packages = ['streamlit', 'pandas', 'numpy', 'plotly']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            issues.append(f"❌ {package} not installed")
    
    # Summary
    print("\n" + "="*50)
    if not issues:
        print("🎉 ALL CHECKS PASSED - Dashboard ready to run!")
        print("🚀 Run: streamlit run renewable_forecast_dashboard.py")
        print("🌐 Access: http://localhost:8501")
    else:
        print("⚠️ ISSUES FOUND:")
        for issue in issues:
            print(f"   {issue}")
        
        if any("❌" in issue for issue in issues):
            print("\n🛠️ Critical issues found. Please resolve before running dashboard.")
            return False
        else:
            print("\n✅ Minor issues only. Dashboard should still work.")
            return True
    
    return True

if __name__ == "__main__":
    success = validate_dashboard_requirements()
    sys.exit(0 if success else 1)
