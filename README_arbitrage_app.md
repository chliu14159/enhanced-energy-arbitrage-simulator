# Jiangsu Energy Arbitrage Dashboard

## Overview
Interactive Streamlit application showcasing profitable arbitrage strategies in the Jiangsu electricity market. This concept-level demo visualizes how energy retailers can achieve 18-25% EBITDA margins through systematic arbitrage opportunities.

## Features

### 🎯 Six Arbitrage Strategies
1. **Temporal Arbitrage** - Contract vs spot price differences (¥50-200/MWh)
2. **Zonal Arbitrage** - Geographic price spreads North/South Jiangsu (¥25-75/MWh)
3. **Renewable Forecast** - Superior wind/solar predictions (¥30-100/MWh)
4. **Coal Correlation** - Anticipate generator cost changes (3-7% advantage)
5. **Time-of-Use** - 96-point daily optimization (¥150-250/MWh)
6. **Portfolio Integration** - Combined strategy performance

### 📊 Interactive Visualizations
- Real-time price comparisons and spreads
- Profit opportunity identification
- Risk-return analysis
- Performance metrics and KPIs
- Implementation roadmap

### 🎛️ Configurable Parameters
- Portfolio size and risk tolerance
- Strategy allocation and exposure limits
- Market timing and position sizing
- Customer participation rates

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run arbitrage_app.py
```

## Usage

1. **Select Strategy**: Choose from individual arbitrage strategies or portfolio view
2. **Configure Parameters**: Adjust risk tolerance, portfolio size, and strategy parameters
3. **Analyze Opportunities**: View interactive charts showing profit potential
4. **Monitor Performance**: Track KPIs and cumulative returns
5. **Plan Implementation**: Follow the phased roadmap for deployment

## Key Insights

### Strategy Performance
- **Temporal Arbitrage**: 15-20 opportunities/month, ¥50-200/MWh profit
- **Zonal Arbitrage**: Daily opportunities, 40-60% profitable days
- **Renewable Forecast**: 8-12 trades/month, 70-80% success rate
- **Coal Correlation**: 2-4 month timing advantage, 3-7% cost savings
- **Time-of-Use**: Daily optimization, 200-400 ¥/MWh peak-valley spread

### Portfolio Targets
- **Conservative**: 15-18% EBITDA margin
- **Moderate**: 18-22% EBITDA margin  
- **Aggressive**: 22-25% EBITDA margin

## Technical Architecture

### Data Sources
- Mock historical price data (2024)
- Realistic market volatility patterns
- Seasonal and daily price variations
- Coal price correlations

### Calculations
- Contract for differences settlement
- Zonal price spread analysis  
- Renewable forecast error impact
- Coal price leading indicators
- 96-point intraday optimization

### Risk Management
- Deviation tolerance monitoring (±3%)
- Portfolio exposure limits
- Correlation analysis
- Stress testing scenarios

## Business Application

This demo shows how energy retailers can:
1. **Identify Opportunities**: Systematic approach to finding profitable trades
2. **Quantify Returns**: Data-driven profit calculations and projections
3. **Manage Risk**: Comprehensive risk assessment and mitigation
4. **Optimize Operations**: Technology-driven trading strategies
5. **Scale Business**: Path from 10% to 25% EBITDA margins

## Next Steps

1. **Real Data Integration**: Connect to actual Jiangsu market data feeds
2. **Machine Learning**: Implement forecasting models for renewable generation
3. **Automated Trading**: Build algorithmic trading systems
4. **Customer Platform**: Deploy demand response management systems
5. **Regulatory Integration**: Ensure compliance with evolving market rules

The application demonstrates the transformation from traditional energy retail (low margins) to sophisticated trading operations (high margins) through systematic arbitrage strategies.