# Energy Price EDA Dashboard

## 📊 Overview
Comprehensive Exploratory Data Analysis dashboard for energy price prediction modeling using Jiangsu Province electricity market data (July 2025).

## 🚀 Quick Start

### 1. Data Processing
```bash
# First, clean and structure the raw Excel data
python data_processor.py
```

### 2. Launch EDA Dashboard
```bash
# Start the Streamlit dashboard
streamlit run streamlit_eda_dashboard.py
```

### 3. Quick Analysis Summary
```bash
# Get a quick text summary of key findings
python test_eda_analysis.py
```

## 📈 Dashboard Features

### Navigation Sections:
1. **Overview & Summary** - Dataset statistics and sample data
2. **Time Series Analysis** - Price trends and temporal patterns  
3. **Daily Patterns** - Hourly cycles and weekday/weekend effects
4. **Correlation Analysis** - Relationships between market factors
5. **Distribution Analysis** - Price distributions and scatter plots
6. **Price Insights** - Key findings and modeling recommendations

### Key Visualizations:
- Day-ahead vs Real-time price comparison
- Price spread volatility analysis
- Hourly price patterns and volatility
- Load vs renewable generation impact
- Correlation heatmaps
- Price distribution histograms

## 🔍 Key Findings

### Price Characteristics:
- **Average Day-ahead Price**: 306.39 ± 155.36 RMB/MWh
- **Average Real-time Price**: 322.49 ± 159.71 RMB/MWh  
- **Strong DA-RT Correlation**: 0.879 (excellent predictor)
- **High Volatility**: σ = 159.71 RMB/MWh

### Market Patterns:
- **Inverted Peak Pricing**: Off-peak hours (355 RMB/MWh) > Peak hours (295 RMB/MWh)
- **Weekend Discount**: 37 RMB/MWh lower than weekdays
- **Renewable Impact**: Strong negative correlation (-0.621) with prices
- **Thermal Capacity**: Positive correlation (0.690) with prices

### Most Volatile Times:
- **Peak Volatility**: 20:00 (σ = 162.30)
- **Lowest Volatility**: 05:00 (σ = 64.70)

## 🎯 Modeling Recommendations

### 1. **Primary Target**: Real-time Price Prediction
- Strong baseline using day-ahead prices (r=0.879)
- Include renewable generation forecasts (strong negative impact)
- Add thermal capacity bidding space (strong positive correlation)

### 2. **Feature Engineering Priorities**:
```python
# Core features
- 日前出清电价 (Day-ahead price) - Primary predictor
- 新能源预测 (Renewable forecast) - Strong negative correlation  
- 竞价空间(火电) (Thermal bidding space) - Strong positive correlation
- hour, is_peak, is_weekend - Temporal patterns

# Advanced features  
- Price lags (1-4 periods)
- Moving averages (24h, 48h)
- Renewable penetration ratio
- Load-generation balance
```

### 3. **Model Architecture Suggestions**:
- **Linear Baseline**: Ridge/Lasso regression with temporal features
- **Time Series**: ARIMA with external regressors (ARIMAX)
- **Machine Learning**: Random Forest, XGBoost with lag features
- **Deep Learning**: LSTM for complex temporal dependencies

### 4. **Validation Strategy**:
- Time-based cross-validation (no data leakage)
- Focus on 15-minute ahead prediction horizon
- Evaluate using MAPE (business-relevant metric)

### 5. **Business Considerations**:
- Inverted peak pricing suggests renewable oversupply during day
- High volatility requires confidence intervals
- Weekend patterns differ significantly from weekdays
- Extreme price events (1-959 RMB/MWh range) need special handling

## 📁 File Structure
```
energy_trading_js/
├── input/july2025.xlsx                 # Raw Excel data
├── cleaned_data/
│   ├── energy_data_cleaned.csv         # Processed dataset  
│   ├── energy_data_cleaned.parquet     # Compressed format
│   └── data_dictionary.txt             # Feature documentation
├── data_processor.py                   # Data cleaning script
├── streamlit_eda_dashboard.py          # Interactive dashboard
├── test_eda_analysis.py               # Quick analysis summary
└── README_EDA.md                      # This documentation
```

## 🔧 Dependencies
```bash
pip install pandas numpy plotly streamlit seaborn matplotlib openpyxl pyarrow
```

## 🌐 Dashboard Access
- **Local URL**: http://localhost:8502
- **Default Port**: 8502 (configurable)

## 📋 Next Steps
1. **Feature Engineering**: Create lagged variables and interaction terms
2. **Model Development**: Implement baseline and advanced models
3. **Validation Framework**: Setup time-based cross-validation
4. **Production Pipeline**: Automate data ingestion and prediction
5. **Business Integration**: Connect with existing arbitrage simulator