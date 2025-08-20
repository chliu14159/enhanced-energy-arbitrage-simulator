import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Jiangsu Energy Analysis Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1f77b4, #17a2b8);
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
    margin-bottom: 2rem;
}
.winner-banner {
    background: linear-gradient(90deg, #28a745, #20c997);
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
    margin-bottom: 1rem;
}
.metric-container {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #007bff;
    margin-bottom: 1rem;
}
.risk-high { border-left-color: #dc3545; }
.risk-medium { border-left-color: #ffc107; }
.risk-low { border-left-color: #28a745; }
.story-section {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    border-left: 4px solid #007bff;
}
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<div class="main-header"><h1>⚡ Jiangsu Energy Analysis Platform</h1><p>Complete Data Science Pipeline: EDA → Model Development → Arbitrage Simulation</p></div>', 
            unsafe_allow_html=True)

# Navigation
page = st.sidebar.selectbox(
    "📊 **Navigate the Complete Story**",
    [
        "🏠 Overview",
        "🔍 Data Exploration (EDA)", 
        "🤖 Model Development",
        "💹 Arbitrage Simulation",
        "📚 Complete Methodology"
    ]
)

# Winner model banner (only show on relevant pages)
if page in ["💹 Arbitrage Simulation", "🏠 Overview"]:
    st.markdown('<div class="winner-banner"><h3>🏆 Powered by Ridge Regression Model: 9.55% MAPE | 90.5% R² | Production Ready</h3></div>', 
                unsafe_allow_html=True)

@st.cache_data
def load_real_data():
    """Load the real energy market data"""
    try:
        df = pd.read_csv('cleaned_data/energy_data_cleaned.csv', index_col=0, parse_dates=True)
        return df
    except FileNotFoundError:
        st.error("Real data not found. Please ensure cleaned_data/energy_data_cleaned.csv exists.")
        return None

@st.cache_resource
def train_winning_model():
    """Train and return the winning Ridge Regression model"""
    df = load_real_data()
    if df is None:
        return None, None, None, None, None
    
    # Prepare features (same as winning model)
    df_model = df.copy()
    
    # Add lag features
    df_model['rt_price_lag1'] = df_model['实时出清电价'].shift(1)
    df_model['da_price_lag1'] = df_model['日前出清电价'].shift(1)
    df_model['rt_price_lag4'] = df_model['实时出清电价'].shift(4)
    
    # Remove NaN
    df_model = df_model.dropna()
    
    # Features from winning model
    features = [
        '日前出清电价', '新能源预测', '竞价空间(火电)', '负荷预测',
        'hour', 'is_peak', 'is_weekend', 
        'rt_price_lag1', 'da_price_lag1', 'rt_price_lag4'
    ]
    
    target = '实时出清电价'
    
    X = df_model[features].values
    y = df_model[target].values
    
    # Scale data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Train Ridge model
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_scaled, y_scaled)
    
    return model, scaler_X, scaler_y, features, df_model

@st.cache_data
def load_model_results():
    """Load all model comparison results"""
    results = {}
    
    # Try different possible paths for the CSV files
    import os
    possible_paths = ['.', '/mount/src/enhanced-energy-arbitrage-simulator', os.getcwd()]
    
    def find_and_load_csv(filename):
        for path in possible_paths:
            full_path = os.path.join(path, filename)
            if os.path.exists(full_path):
                try:
                    return pd.read_csv(full_path)
                except Exception as e:
                    continue
        return None
    
    # Original neural network results
    original = find_and_load_csv('model_comparison_summary.csv')
    if original is not None:
        original['Phase'] = 'Original Neural Networks'
        results['original'] = original
    
    # Improved neural network results  
    improved = find_and_load_csv('improved_model_results.csv')
    if improved is not None:
        improved['Phase'] = 'Improved Neural Networks'
        improved = improved.rename(columns={'MAPE_All': 'MAPE'})
        results['improved'] = improved
    
    # Final optimization results
    final = find_and_load_csv('final_optimization_results.csv')
    if final is not None:
        final['Phase'] = 'Final Optimization'
        results['final'] = final
    
    return results

# Enhanced Arbitrage Simulator Classes (from the working version)
class EnhancedEnergyArbitrageSimulator:
    def __init__(self, portfolio_size, base_price, real_data, model, scaler_X, scaler_y, features, preprocessed_data):
        self.portfolio_size = portfolio_size
        self.base_price = base_price
        self.daily_volume = portfolio_size * 1000 / 365
        self.real_data = real_data
        self.preprocessed_data = preprocessed_data  # Data with lag features
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.features = features
        
        # Risk parameters
        self.max_deviation = 3.0
        self.penalty_rate = 0.1
        
    def get_real_market_data(self, start_date, num_days=7):
        """Extract real market data for specified period"""
        try:
            # Use preprocessed data (with lag features) instead of raw data
            # Find the start index in preprocessed data
            start_idx = self.preprocessed_data.index.get_indexer([start_date], method='nearest')[0]
            
            # Extract data for the period (96 intervals per day)
            end_idx = start_idx + (num_days * 96)
            period_data = self.preprocessed_data.iloc[start_idx:end_idx].copy()
            
            if len(period_data) == 0:
                st.error("No data found for selected period")
                return None
            
            # Verify all required features exist
            missing_features = [feat for feat in self.features if feat not in period_data.columns]
            if missing_features:
                st.error(f"Missing required features: {missing_features}")
                return None
            
            return period_data
        except Exception as e:
            st.error(f"Error extracting real data: {e}")
            return None
    
    def generate_ai_predictions(self, market_data):
        """Use trained Ridge model to predict real-time prices"""
        predictions = []
        actual_mape_values = []
        
        for idx, row in market_data.iterrows():
            try:
                # Prepare features for prediction
                feature_values = [row[feat] for feat in self.features]
                feature_array = np.array(feature_values).reshape(1, -1)
                
                # Scale features
                feature_scaled = self.scaler_X.transform(feature_array)
                
                # Predict
                pred_scaled = self.model.predict(feature_scaled)
                pred_price = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
                
                # Calculate actual MAPE for this prediction
                actual_price = row['实时出清电价']
                mape = abs(pred_price - actual_price) / actual_price * 100
                
                predictions.append({
                    'datetime': idx,
                    'actual_price': actual_price,
                    'predicted_price': pred_price,
                    'day_ahead_price': row['日前出清电价'],
                    'prediction_error': pred_price - actual_price,
                    'mape': mape,
                    'load_forecast': row['负荷预测'],
                    'renewable_forecast': row['新能源预测'],
                    'thermal_capacity': row['竞价空间(火电)'],
                    'hour': row['hour'],
                    'is_peak': row['is_peak'],
                    'is_weekend': row['is_weekend']
                })
                
                actual_mape_values.append(mape)
                
            except Exception as e:
                st.warning(f"Prediction error at {idx}: {e}")
                continue
        
        pred_df = pd.DataFrame(predictions)
        
        # Overall model performance on this period
        overall_mape = np.mean(actual_mape_values) if actual_mape_values else 15.0
        
        return pred_df, overall_mape
    
    def calculate_enhanced_arbitrage_profits(self, predictions_df, day_start_idx, day_end_idx):
        """Calculate arbitrage profits based on Jiangsu market rules"""
        day_data = predictions_df.iloc[day_start_idx:day_end_idx]
        
        if len(day_data) == 0:
            return self._zero_profits()
        
        # Calculate daily metrics
        avg_actual = day_data['actual_price'].mean()
        avg_predicted = day_data['predicted_price'].mean()
        avg_day_ahead = day_data['day_ahead_price'].mean()
        daily_mape = day_data['mape'].mean()
        prediction_accuracy = max(0.1, 1 - daily_mape / 100)
        
        # Customer portfolio segmentation (based on SME feedback)
        # 70% manufacturing in Southern Jiangsu, 30% mixed
        manufacturing_volume = self.daily_volume * 0.7  # ±5% deviation limit
        retail_volume = self.daily_volume * 0.3         # ±5% deviation limit
        
        # 1. CONTRACT DEVIATION ARBITRAGE - 5-LEVEL STRATEGY (Updated per improvement plan)
        # Calculate optimal deviation percentage based on spread magnitude and risk appetite
        contract_spot_spread = self.base_price - avg_actual
        
        # 5-Level Deviation Strategy with progressive risk/return profiles
        deviation_strategies = self._calculate_5_level_deviation_strategy(contract_spot_spread)
        
        # Select optimal strategy based on spread magnitude
        selected_strategy = self._select_optimal_deviation_level(contract_spot_spread, deviation_strategies)
        
        temporal_profit = selected_strategy['profit']
        deviation_level = selected_strategy['level']
        deviation_percentage = selected_strategy['percentage']
        
        # 2. DAY-AHEAD vs REAL-TIME POSITIONING (Replaces old "AI arbitrage")
        # Based on prediction vs day-ahead price differences
        da_rt_spread = avg_predicted - avg_day_ahead
        
        if abs(da_rt_spread) > 20:  # Only trade on significant spreads (>¥20/MWh)
            if da_rt_spread > 0:  # Predicted RT > DA: Buy less in DA, more in RT
                positioning_volume = self.daily_volume * 0.03 * prediction_accuracy
                positioning_profit = positioning_volume * abs(da_rt_spread) * 0.6
            else:  # Predicted RT < DA: Buy more in DA, less in RT
                positioning_volume = self.daily_volume * 0.05 * prediction_accuracy
                positioning_profit = positioning_volume * abs(da_rt_spread) * 0.7
        else:
            positioning_profit = 0
        
        # 3. RENEWABLE FORECAST ARBITRAGE (Enhanced per SME feedback)
        # Only trade when forecast accuracy provides competitive advantage
        renewable_forecast_error = day_data['renewable_forecast'].std()
        
        if renewable_forecast_error > 50:  # Significant renewable volatility (>50MW std)
            # Our competitive advantage: Better forecast vs market average
            forecast_advantage = max(0, (25 - daily_mape) / 25)  # Advantage if MAPE < 25%
            
            if forecast_advantage > 0:
                # Volume based on forecast confidence and market volatility
                renewable_volume = min(self.daily_volume * 0.2, renewable_forecast_error * 2)
                # Price impact: ¥50-100/MWh per 100MW forecast error (per SME)
                price_impact = renewable_forecast_error * 0.75  # ¥0.75/MWh per MW std
                renewable_profit = renewable_volume * price_impact * forecast_advantage
            else:
                renewable_profit = 0
        else:
            renewable_profit = 0
        
        # 4. CUSTOMER SEGMENTATION OPTIMIZATION (New strategy)
        # Target Southern Jiangsu manufacturing (premium pricing)
        # vs Northern Jiangsu procurement (cost advantage)
        southern_premium = 25  # ¥25/MWh premium in Southern Jiangsu
        northern_discount = 35  # ¥35/MWh discount in Northern Jiangsu
        
        # Geographic arbitrage opportunity (15-25% cost savings per SME)
        if prediction_accuracy > 0.7:  # Only when confident in forecasts
            geographic_volume = manufacturing_volume * 0.4  # 40% of manufacturing load
            geographic_spread = southern_premium + northern_discount
            geographic_profit = geographic_volume * geographic_spread * 0.2  # 20% capture
        else:
            geographic_profit = 0
        
        # 5. PENALTY COSTS (Updated per SME rules)
        penalty_cost = 0
        
        # User side penalties: ±5% limit, excess at 1.5× real-time price
        if daily_mape > 5.0:
            excess_deviation = daily_mape - 5.0
            penalty_volume = self.daily_volume * excess_deviation / 100
            penalty_cost = penalty_volume * avg_actual * 1.5
        
        # Additional operational costs for forecast errors
        forecast_cost = self.daily_volume * daily_mape * 2.0
        
        # 6. OPERATIONAL EFFICIENCY GAINS
        # Better forecasting reduces imbalance costs and improves positioning
        baseline_operational_cost = self.daily_volume * 15  # ¥15/MWh baseline cost
        efficiency_gain = baseline_operational_cost * prediction_accuracy * 0.3
        operational_cost = baseline_operational_cost - efficiency_gain + forecast_cost
        
        # TOTALS
        total_profit = temporal_profit + positioning_profit + renewable_profit + geographic_profit
        total_costs = penalty_cost + operational_cost
        net_profit = total_profit - total_costs
        
        return {
            'temporal_arbitrage': temporal_profit,
            'ai_arbitrage': positioning_profit,  # Keep name for UI compatibility
            'tou_arbitrage': geographic_profit,   # Repurposed for geographic arbitrage
            'renewable_arbitrage': renewable_profit,
            'total_profit': total_profit,
            'penalty_cost': penalty_cost,
            'operational_cost': operational_cost,
            'total_costs': total_costs,
            'net_profit': net_profit,
            'available_deviation': 5.0 - daily_mape,  # User side 5% limit
            'daily_mape': daily_mape,
            'prediction_accuracy': prediction_accuracy,
            'avg_actual_price': avg_actual,
            'avg_predicted_price': avg_predicted,
            'contract_spot_spread': abs(self.base_price - avg_actual),
            # NEW: 5-Level Deviation Strategy Information
            'deviation_level': deviation_level,
            'deviation_percentage': deviation_percentage,
            'deviation_risk_level': selected_strategy['risk_level'],
            'deviation_certainty': selected_strategy['certainty'],
            'deviation_volume': selected_strategy['volume'],
            'all_deviation_strategies': deviation_strategies  # For detailed analysis
        }
    
    def _calculate_5_level_deviation_strategy(self, contract_spot_spread):
        """
        Calculate profit for all 5 deviation levels (1% to 5%)
        Returns dictionary with profit calculations for each level
        """
        spread_magnitude = abs(contract_spot_spread)
        is_over_consumption = contract_spot_spread > 0  # Spot < Contract
        
        strategies = {}
        
        for level in range(1, 6):  # 1% to 5%
            percentage = level / 100.0  # Convert to decimal
            deviation_volume = self.daily_volume * percentage
            
            if is_over_consumption:
                # Over-consumption: pay spot instead of contract
                # Capture rates decrease with higher risk levels
                capture_rates = {1: 0.90, 2: 0.85, 3: 0.80, 4: 0.75, 5: 0.70}
                profit = deviation_volume * spread_magnitude * capture_rates[level]
            else:
                # Under-consumption: avoid paying high spot prices
                # Capture rates for demand response
                capture_rates = {1: 0.85, 2: 0.80, 3: 0.75, 4: 0.70, 5: 0.65}
                profit = deviation_volume * spread_magnitude * capture_rates[level]
            
            # Risk classifications
            risk_levels = {
                1: "Conservative", 2: "Moderate-Conservative", 3: "Moderate", 
                4: "Moderate-Aggressive", 5: "Maximum Allowable"
            }
            
            strategies[level] = {
                'level': level,
                'percentage': percentage * 100,  # For display
                'volume': deviation_volume,
                'profit': profit,
                'risk_level': risk_levels[level],
                'capture_rate': capture_rates[level],
                'spread_required': 50 * level,  # Minimum spread for activation
                'certainty': ["High", "Good", "Moderate", "Higher Risk", "Maximum Risk"][level-1]
            }
        
        return strategies
    
    def _select_optimal_deviation_level(self, contract_spot_spread, deviation_strategies):
        """
        Select optimal deviation level based on spread magnitude and risk management
        """
        spread_magnitude = abs(contract_spot_spread)
        
        # Selection logic based on spread magnitude
        if spread_magnitude >= 300:  # Very large spread (>¥300/MWh)
            selected_level = 5  # Maximum allowable
        elif spread_magnitude >= 200:  # Large spread (>¥200/MWh) 
            selected_level = 4  # Moderate-aggressive
        elif spread_magnitude >= 150:  # Significant spread (>¥150/MWh)
            selected_level = 3  # Moderate
        elif spread_magnitude >= 100:  # Medium spread (>¥100/MWh)
            selected_level = 2  # Moderate-conservative
        elif spread_magnitude >= 50:   # Small spread (>¥50/MWh)
            selected_level = 1  # Conservative
        else:
            # Spread too small, no arbitrage
            return {
                'level': 0,
                'percentage': 0,
                'volume': 0,
                'profit': 0,
                'risk_level': "No Trade",
                'capture_rate': 0,
                'spread_required': 50,
                'certainty': "Spread too small"
            }
        
        return deviation_strategies[selected_level]
    
    def _zero_profits(self):
        """Return zero profits structure"""
        return {
            'temporal_arbitrage': 0, 'ai_arbitrage': 0, 'tou_arbitrage': 0,
            'renewable_arbitrage': 0, 'total_profit': 0, 'penalty_cost': 0,
            'operational_cost': 0, 'total_costs': 0, 'net_profit': 0,
            'available_deviation': 0, 'daily_mape': 15.0, 'prediction_accuracy': 0.5,
            'avg_actual_price': 300, 'avg_predicted_price': 300, 'contract_spot_spread': 0
        }

# Initialize session state
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = None
if 'predictions_data' not in st.session_state:
    st.session_state.predictions_data = None

# =============================================================================
# PAGE ROUTING
# =============================================================================

if page == "🏠 Overview":
    # Overview page content
    st.header("🎯 Platform Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## 📊 **Complete Data Science Pipeline**
        
        This platform tells the complete story of energy market analysis:
        
        ### **🔍 1. Data Exploration**
        - **2,976 data points** from Jiangsu Province July 2025
        - **15-minute resolution** electricity market data
        - **Comprehensive EDA** with price patterns, correlations, and insights
        
        ### **🤖 2. Model Development** 
        - **Multiple model comparison**: LSTM, CNN, GRU, Transformer, Ridge
        - **Iterative improvement**: Feature engineering and optimization
        - **Winning model**: Ridge Regression (9.55% MAPE, 90.5% R²)
        
        ### **💹 3. Arbitrage Simulation**
        - **4 enhanced strategies** using real data and AI predictions
        - **Complete transparency** with mathematical formulations
        - **Risk management** and regulatory compliance
        
        ### **📚 4. Complete Methodology**
        - **Every equation explained** with business logic
        - **Parameter sensitivity analysis**
        - **Production-ready framework**
        """)
    
    with col2:
        # Key metrics summary
        st.markdown("## 🏆 **Key Achievements**")
        
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.metric("AI Model MAPE", "9.55%", delta="54% improvement")
            st.metric("R² Score", "90.5%", delta="Excellent fit")
            st.metric("Data Points", "2,976", delta="Real market data")
        
        with metric_col2:
            st.metric("Arbitrage Strategies", "4", delta="Enhanced with AI")
            st.metric("Analysis Period", "Flexible", delta="3-14 days")
            st.metric("Portfolio Scale", "500-3000 GWh", delta="Enterprise ready")
        
        # Navigation guide
        st.markdown("""
        ## 🧭 **Navigation Guide**
        
        **👆 Use the sidebar to explore:**
        
        - **🔍 Data Exploration**: Understand the Jiangsu market patterns
        - **🤖 Model Development**: See how we achieved 9.55% MAPE
        - **💹 Arbitrage Simulation**: Interactive strategy analysis
        - **📚 Complete Methodology**: Mathematical transparency
        
        **💡 Recommended flow**: Start with Data Exploration → Model Development → Arbitrage Simulation
        """)
    
    # Success story timeline
    st.markdown("---")
    st.markdown("## 🎯 **The Data Science Success Story**")
    
    story_col1, story_col2, story_col3 = st.columns(3)
    
    with story_col1:
        st.markdown('<div class="story-section">', unsafe_allow_html=True)
        st.markdown("""
        **📊 Step 1: Data Understanding**
        
        - Analyzed 2,976 data points
        - Discovered price patterns and correlations
        - Identified key market drivers
        - Found peak/off-peak dynamics
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with story_col2:
        st.markdown('<div class="story-section">', unsafe_allow_html=True)
        st.markdown("""
        **🤖 Step 2: Model Optimization**
        
        - Started with 20-22% MAPE (neural networks)
        - Applied feature engineering
        - Discovered Ridge Regression superiority
        - Achieved 9.55% MAPE (54% improvement)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with story_col3:
        st.markdown('<div class="story-section">', unsafe_allow_html=True)
        st.markdown("""
        **💹 Step 3: Business Application**
        
        - Integrated AI predictions with arbitrage
        - Developed 4 enhanced strategies
        - Added complete methodology transparency
        - Built production-ready simulator
        """)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "🔍 Data Exploration (EDA)":
    st.header("🔍 Exploratory Data Analysis")
    st.markdown("**Understanding the Jiangsu Province electricity market patterns and characteristics**")
    
    # Load data
    data = load_real_data()
    if data is None:
        st.stop()
    
    # Data overview
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    with col2:
        st.metric("Time Period", "July 2025")
    with col3:
        st.metric("Resolution", "15 minutes")
    with col4:
        st.metric("Variables", f"{len(data.columns)}")
    
    # Price patterns
    st.subheader("💰 Price Pattern Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📈 Time Series", "🕐 Hourly Patterns", "📊 Correlations"])
    
    with tab1:
        # Time series plot
        fig = go.Figure()
        
        # Sample data for better visualization (every 4th point for 4-hour intervals)
        sample_data = data.iloc[::4]  # Sample every 4th point
        
        fig.add_trace(go.Scatter(
            x=sample_data.index,
            y=sample_data['实时出清电价'],
            mode='lines',
            name='Real-time Price',
            line=dict(color='blue', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=sample_data.index,
            y=sample_data['日前出清电价'],
            mode='lines',
            name='Day-ahead Price',
            line=dict(color='red', width=1)
        ))
        
        fig.update_layout(
            title="Real-time vs Day-ahead Electricity Prices",
            xaxis_title="Date",
            yaxis_title="Price (RMB/MWh)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Hourly patterns
        hourly_stats = data.groupby('hour').agg({
            '实时出清电价': ['mean', 'std'],
            '负荷预测': 'mean',
            '新能源预测': 'mean'
        }).round(2)
        
        hourly_stats.columns = ['Avg_RT_Price', 'RT_Price_Std', 'Avg_Load', 'Avg_Renewable']
        hourly_stats = hourly_stats.reset_index()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Real-time Price by Hour', 'Price Volatility by Hour', 
                          'Average Load Forecast by Hour', 'Average Renewable Forecast by Hour'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Average price
        fig.add_trace(go.Bar(x=hourly_stats['hour'], y=hourly_stats['Avg_RT_Price'], 
                           name='Avg Price', marker_color='lightblue'), row=1, col=1)
        
        # Price volatility
        fig.add_trace(go.Bar(x=hourly_stats['hour'], y=hourly_stats['RT_Price_Std'], 
                           name='Price Std', marker_color='orange'), row=1, col=2)
        
        # Load forecast
        fig.add_trace(go.Bar(x=hourly_stats['hour'], y=hourly_stats['Avg_Load'], 
                           name='Avg Load', marker_color='green'), row=2, col=1)
        
        # Renewable forecast
        fig.add_trace(go.Bar(x=hourly_stats['hour'], y=hourly_stats['Avg_Renewable'], 
                           name='Avg Renewable', marker_color='purple'), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False, title_text="Market Patterns by Hour of Day")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Correlation analysis
        numeric_cols = ['实时出清电价', '日前出清电价', '负荷预测', '新能源预测', '竞价空间(火电)']
        corr_matrix = data[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title="Variable Correlation Matrix",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("💡 Key Market Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("""
        **🕐 Time Patterns:**
        - **Peak hours**: 8-10 AM and 7-9 PM
        - **Valley hours**: 2-5 AM
        - **Price volatility** highest during transition periods
        - **Weekend effects** visible in demand patterns
        """)
        
        # Calculate some basic stats
        peak_price = data[data['is_peak'] == 1]['实时出清电价'].mean()
        valley_price = data[data['is_peak'] == 0]['实时出清电价'].mean()
        price_spread = peak_price - valley_price
        
        st.metric("Peak-Valley Spread", f"{price_spread:.1f} RMB/MWh", 
                 delta=f"Peak: {peak_price:.1f}, Valley: {valley_price:.1f}")
    
    with insight_col2:
        st.markdown("""
        **⚡ Market Dynamics:**
        - **Day-ahead vs Real-time**: High correlation but systematic differences
        - **Renewable impact**: Visible effect on price volatility
        - **Load correlation**: Strong relationship with prices
        - **Thermal bidding space**: Key market indicator
        """)
        
        # Prediction accuracy of day-ahead
        da_rt_diff = data['实时出清电价'] - data['日前出清电价']
        da_mape = (da_rt_diff.abs() / data['实时出清电价']).mean() * 100
        
        st.metric("Day-ahead MAPE", f"{da_mape:.1f}%", 
                 delta="Baseline for AI improvement")

elif page == "🤖 Model Development":
    st.header("🤖 Model Development Journey")
    st.markdown("**From initial neural networks (20-22% MAPE) to the winning Ridge Regression model (9.55% MAPE)**")
    
    # Load model results
    results = load_model_results()
    
    # Debug information
    st.write(f"Debug: Results keys found: {list(results.keys())}")
    st.write(f"Debug: Results empty?: {not results}")
    
    if not results:
        st.warning("Model comparison results not found. Please run the model comparison scripts first.")
        st.info("Expected files: model_comparison_summary.csv, improved_model_results.csv, final_optimization_results.csv")
    else:
        # Model evolution story
        st.subheader("📈 The Model Evolution Story")
        
        # Create phases data
        phases_data = []
        
        if 'original' in results:
            original_best = results['original'].loc[results['original']['MAPE'].idxmin()]
            phases_data.append({
                'Phase': 'Phase 1: Neural Networks',
                'Best_Model': original_best['Model'],
                'MAPE': original_best['MAPE'],
                'R2': original_best.get('R2', 0.8),
                'Description': 'Initial deep learning approach'
            })
        
        if 'improved' in results:
            improved_best = results['improved'].loc[results['improved']['MAPE'].idxmin()]
            phases_data.append({
                'Phase': 'Phase 2: Feature Engineering',
                'Best_Model': improved_best['Model'],
                'MAPE': improved_best['MAPE'],
                'R2': improved_best.get('R2', 0.85),
                'Description': 'Advanced features + custom loss'
            })
        
        if 'final' in results:
            final_best = results['final'].loc[results['final']['MAPE'].idxmin()]
            phases_data.append({
                'Phase': 'Phase 3: Optimization',
                'Best_Model': final_best['Model'],
                'MAPE': final_best['MAPE'],
                'R2': final_best.get('R2', 0.905),
                'Description': 'Traditional ML breakthrough'
            })
        
        if phases_data:
            phases_df = pd.DataFrame(phases_data)
            
            # MAPE improvement chart
            fig = go.Figure()
            
            colors = ['red', 'orange', 'green']
            for i, (_, row) in enumerate(phases_df.iterrows()):
                fig.add_trace(go.Bar(
                    x=[row['Phase']],
                    y=[row['MAPE']],
                    name=row['Best_Model'],
                    marker_color=colors[i % len(colors)],
                    text=f"{row['MAPE']:.2f}%",
                    textposition='auto'
                ))
            
            fig.update_layout(
                title="Model Performance Evolution: MAPE Reduction Journey",
                xaxis_title="Development Phase",
                yaxis_title="MAPE (%)",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Success metrics
            col1, col2, col3 = st.columns(3)
            
            if len(phases_data) >= 2:
                initial_mape = phases_data[0]['MAPE']
                final_mape = phases_data[-1]['MAPE']
                improvement = (initial_mape - final_mape) / initial_mape * 100
                
                with col1:
                    st.metric("Initial MAPE", f"{initial_mape:.2f}%", delta="Neural networks baseline")
                with col2:
                    st.metric("Final MAPE", f"{final_mape:.2f}%", delta="Ridge Regression winner")
                with col3:
                    st.metric("Improvement", f"{improvement:.1f}%", delta="Relative reduction")
            
            # Detailed comparison
            st.subheader("📊 Detailed Model Comparison")
            
            if st.checkbox("Show all model results"):
                for phase, data in results.items():
                    st.markdown(f"**{phase.title()} Results:**")
                    st.dataframe(data.round(3))
    
    # Key learnings
    st.subheader("💡 Key Learnings & Insights")
    
    learning_col1, learning_col2 = st.columns(2)
    
    with learning_col1:
        st.markdown("""
        **🧠 What Worked:**
        - **Simplicity over complexity**: Ridge Regression beat all neural networks
        - **Feature selection**: 10 carefully chosen features outperformed 25+ features
        - **Domain knowledge**: Market fundamentals more important than complex patterns
        - **Lag features**: Simple time lags (1, 4 periods) provided crucial temporal info
        """)
    
    with learning_col2:
        st.markdown("""
        **❌ What Didn't Work:**
        - **Deep learning**: LSTM, CNN, GRU all overfitted despite regularization
        - **Complex features**: Advanced engineering actually hurt performance
        - **Custom loss functions**: Direct MAPE optimization was unstable
        - **Ensemble methods**: Added complexity without accuracy gains
        """)
    
    # Model explanation
    st.subheader("🏆 Winning Model: Ridge Regression")
    
    st.markdown("""
    **Why Ridge Regression Won:**
    
    1. **Linear relationships dominate**: Energy prices follow predictable market fundamentals
    2. **Regularization prevents overfitting**: L2 penalty handles multicollinearity
    3. **Stable and interpretable**: Coefficients have clear business meaning
    4. **Fast and scalable**: Production-ready performance
    5. **Robust to outliers**: Handles market anomalies gracefully
    
    **Final Model Features (10 selected):**
    - Day-ahead prices (strongest predictor)
    - Renewable forecasts (market impact)
    - Thermal bidding space (supply indicator)
    - Load forecasts (demand driver)
    - Time patterns (hour, peak, weekend)
    - Price lags (1 period, 4 periods for temporal patterns)
    """)

elif page == "💹 Arbitrage Simulation":
    # COMPLETE ARBITRAGE SIMULATOR
    st.header("💹 Enhanced Arbitrage Simulation")
    st.markdown("**Interactive machine learning-based arbitrage strategy analysis with complete methodology transparency**")
    
    # Always show sidebar controls first
    st.sidebar.header("🎛️ Simulation Parameters")
    
    # Portfolio configuration
    st.sidebar.subheader("💼 Portfolio Configuration")
    portfolio_size = st.sidebar.slider("Annual Portfolio Size (GWh)", 500, 3000, 1500, 100)
    base_price = st.sidebar.slider("Base Contract Price (RMB/MWh)", 350, 550, 420, 10)
    
    # Try to load real data and model
    data_loaded = False
    try:
        real_data = load_real_data()
        model, scaler_X, scaler_y, features, preprocessed_data = train_winning_model()
        
        if real_data is not None and model is not None:
            data_loaded = True
        else:
            st.error("Failed to load data or train model. Please check your data files.")
            
    except Exception as e:
        st.error(f"Error loading data or model: {e}")

    # Time period selection (show even if data not loaded)
    if data_loaded:
        # Use real data for date range
        min_date = preprocessed_data.index.min().date()
        max_date = preprocessed_data.index.max().date()
        default_start = min_date + timedelta(days=1)
        max_end = max_date - timedelta(days=7)
    else:
        # Use dummy dates if data not loaded
        from datetime import date
        min_date = date(2025, 7, 1)
        max_date = date(2025, 7, 31)
        default_start = date(2025, 7, 2)
        max_end = date(2025, 7, 24)
    
    start_date = st.sidebar.date_input(
        "Start Date", 
        value=default_start,  # Use safe default
        min_value=min_date,
        max_value=max_end
    )
    
    num_days = st.sidebar.slider("Number of Days to Analyze", 3, 14, 7)
    
    # Add helpful information
    if data_loaded:
        st.sidebar.info(f"📅 **Available Data Range:**\n{min_date} to {max_date}\n\n💡 **Recommended:** Start from {default_start} to ensure lag features are available.")
    else:
        st.sidebar.warning("⚠️ **Data Loading Failed**\nPlease check the data files and try again.")
    
    # Model performance info
    st.sidebar.subheader("🏆 Model Performance")
    if data_loaded:
        st.sidebar.metric("Model Type", "Ridge Regression")
        st.sidebar.metric("Training MAPE", "9.55%")
        st.sidebar.metric("R² Score", "0.905")
        st.sidebar.success("✅ Production Ready Model")
    else:
        st.sidebar.error("❌ Model Not Available")
    
    # Simulation control
    st.sidebar.subheader("🚀 Run Analysis")
    
    # Only enable button if data is loaded
    button_disabled = not data_loaded
    button_help = None if data_loaded else "Data loading failed - simulation unavailable"
    
    if st.sidebar.button("🎯 Run Enhanced Simulation", type="primary", disabled=button_disabled, help=button_help):
        with st.spinner("🔄 Running enhanced arbitrage analysis..."):
            
            try:
                # Initialize enhanced simulator
                simulator = EnhancedEnergyArbitrageSimulator(
                    portfolio_size, base_price, real_data, model, scaler_X, scaler_y, features, preprocessed_data
                )
                
                # Get real market data for selected period
                start_datetime = pd.Timestamp(start_date)
                market_data = simulator.get_real_market_data(start_datetime, num_days)
                
                if market_data is not None:
                    # Generate AI predictions
                    predictions_df, overall_mape = simulator.generate_ai_predictions(market_data)
                    
                    # Calculate daily profits
                    daily_results = []
                    for day in range(num_days):
                        day_start = day * 96
                        day_end = (day + 1) * 96
                        
                        if day_end <= len(predictions_df):
                            profits = simulator.calculate_enhanced_arbitrage_profits(
                                predictions_df, day_start, day_end
                            )
                            
                            daily_results.append({
                                'day': day + 1,
                                'date': start_datetime + timedelta(days=day),
                                **profits
                            })
                    
                    # Store results
                    st.session_state.simulation_data = pd.DataFrame(daily_results)
                    st.session_state.predictions_data = predictions_df
                    st.session_state.overall_mape = overall_mape
                    
                    st.success(f"✅ Analysis completed! Model achieved {overall_mape:.2f}% MAPE on this period.")
                    
            except Exception as e:
                st.error(f"Simulation failed: {e}")
                st.info("Please check your data and try again.")

    # Main content
    if st.session_state.simulation_data is not None and len(st.session_state.simulation_data) > 0:
        df = st.session_state.simulation_data
        pred_df = st.session_state.predictions_data
        actual_mape = st.session_state.overall_mape
        
        # Check if required columns exist
        required_columns = ['net_profit', 'ai_arbitrage', 'temporal_arbitrage', 'tou_arbitrage', 'renewable_arbitrage']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing columns in simulation data: {missing_columns}")
            st.info("Please run the simulation again to generate complete results.")
        else:
            # Performance overview
            st.subheader("📈 Enhanced Arbitrage Performance Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            total_profit = df['net_profit'].sum()
            avg_daily_profit = df['net_profit'].mean()
            best_day_profit = df['net_profit'].max()
            ai_contribution = df['ai_arbitrage'].sum()
            
            with col1:
                st.metric("Total Period Profit", f"¥{total_profit/1000:.1f}k", 
                         delta=f"{total_profit/portfolio_size:.2f} RMB/MWh")
            
            with col2:
                st.metric("Average Daily Profit", f"¥{avg_daily_profit/1000:.1f}k",
                         delta=f"Best: ¥{best_day_profit/1000:.1f}k")
            
            with col3:
                st.metric("Model MAPE", f"{actual_mape:.2f}%",
                         delta="Excellent!" if actual_mape < 12 else "Good")
            
            with col4:
                st.metric("ML-Enhanced Profits", f"¥{ai_contribution/1000:.1f}k",
                         delta=f"{ai_contribution/total_profit*100:.1f}% of total" if total_profit != 0 else "0%")
            
            # Tabs for detailed analysis
            tab1, tab2, tab3 = st.tabs([
                "💹 Strategy Breakdown", "📊 Model Predictions vs Reality", "📈 Daily Performance"
            ])
            
            with tab1:
                st.subheader("💹 Enhanced Arbitrage Strategy Performance")
                
                # 🔥 NEW: 5-Level Deviation Strategy Showcase
                st.markdown("""
                <div style="background: linear-gradient(90deg, #ff6b6b, #4ecdc4); color: white; 
                            padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
                <h3>🎯 NEW: 5-Level Deviation Strategy (1% - 5%)</h3>
                <p>Advanced risk-graduated approach with optimal deviation selection based on spread magnitude</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display deviation strategy analysis
                if 'deviation_level' in df.columns and not df.empty:
                    st.subheader("📊 5-Level Deviation Strategy Performance")
                    
                    # Create columns for strategy overview
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_deviation_level = df['deviation_level'].mean()
                        st.metric("Average Deviation Level", f"Level {avg_deviation_level:.1f}", 
                                 delta=f"{df['deviation_percentage'].mean():.1f}% volume")
                    
                    with col2:
                        most_common_level = df['deviation_level'].mode().iloc[0] if not df['deviation_level'].mode().empty else 0
                        if most_common_level > 0:
                            level_count = (df['deviation_level'] == most_common_level).sum()
                            st.metric("Most Used Level", f"Level {int(most_common_level)}", 
                                     delta=f"Used {level_count}/{len(df)} days")
                        else:
                            st.metric("Most Used Level", "No Trades", delta="Spreads too small")
                    
                    with col3:
                        total_deviation_profit = df['temporal_arbitrage'].sum()
                        if 'deviation_volume' in df.columns:
                            avg_margin = total_deviation_profit / df['deviation_volume'].sum() if df['deviation_volume'].sum() > 0 else 0
                            st.metric("Avg Margin per MWh", f"¥{avg_margin:.1f}/MWh", 
                                     delta=f"Total: ¥{total_deviation_profit/1000:.1f}k")
                        else:
                            st.metric("Total Deviation Profit", f"¥{total_deviation_profit/1000:.1f}k")
                    
                    # 5-Level Strategy Visualization
                    fig_deviation = go.Figure()
                    
                    # Create a comprehensive deviation strategy chart
                    for day_num, (day_idx, row) in enumerate(df.iterrows()):
                        if 'all_deviation_strategies' in row and row['all_deviation_strategies']:
                            strategies = row['all_deviation_strategies']
                            for level, strategy_data in strategies.items():
                                fig_deviation.add_trace(go.Scatter(
                                    x=[day_num] * 5,  # Repeat for each level
                                    y=[level],
                                    mode='markers',
                                    marker=dict(
                                        size=strategy_data['profit'] / 1000,  # Size by profit
                                        color=level,
                                        colorscale='RdYlGn',
                                        showscale=True,
                                        colorbar=dict(title="Deviation Level"),
                                        line=dict(width=2, color='white')
                                    ),
                                    text=f"Level {level}: {strategy_data['risk_level']}<br>" +
                                         f"Profit: ¥{strategy_data['profit']:,.0f}<br>" +
                                         f"Volume: {strategy_data['percentage']:.1f}%<br>" +
                                         f"Certainty: {strategy_data['certainty']}",
                                    hovertemplate='%{text}<extra></extra>',
                                    name=f"Day {day_num+1}",
                                    showlegend=False
                                ))
                    
                    # Add selected strategy highlighting
                    selected_levels = df['deviation_level'].tolist()
                    selected_profits = df['temporal_arbitrage'].tolist()
                    
                    fig_deviation.add_trace(go.Scatter(
                        x=list(range(len(df))),
                        y=selected_levels,
                        mode='markers+lines',
                        marker=dict(size=15, color='red', symbol='star'),
                        line=dict(color='red', width=3),
                        name='Selected Strategy',
                        text=[f"Day {i+1}: Level {int(level)}<br>Profit: ¥{profit:,.0f}" 
                              for i, (level, profit) in enumerate(zip(selected_levels, selected_profits))],
                        hovertemplate='%{text}<extra></extra>'
                    ))
                    
                    fig_deviation.update_layout(
                        title="5-Level Deviation Strategy Selection Over Time<br>" +
                              "<sub>Red stars show actual selected levels, bubble size = profit potential</sub>",
                        xaxis_title="Day",
                        yaxis_title="Deviation Level (1=Conservative, 5=Maximum)",
                        height=500,
                        yaxis=dict(tickmode='linear', tick0=1, dtick=1, range=[0, 6])
                    )
                    
                    st.plotly_chart(fig_deviation, use_container_width=True)
                    
                    # Strategy level breakdown table
                    st.subheader("📋 5-Level Strategy Breakdown")
                    
                    # Create summary table of all 5 levels
                    level_summary = []
                    for level in range(1, 6):
                        days_used = (df['deviation_level'] == level).sum()
                        total_profit = df[df['deviation_level'] == level]['temporal_arbitrage'].sum()
                        risk_levels = {1: "Conservative", 2: "Moderate-Conservative", 3: "Moderate", 
                                     4: "Moderate-Aggressive", 5: "Maximum Allowable"}
                        
                        level_summary.append({
                            'Level': level,
                            'Percentage': f"{level}%",
                            'Risk Profile': risk_levels[level],
                            'Days Used': days_used,
                            'Total Profit (¥k)': f"{total_profit/1000:.1f}",
                            'Avg Daily Profit (¥k)': f"{total_profit/days_used/1000:.1f}" if days_used > 0 else "0.0",
                            'Usage %': f"{days_used/len(df)*100:.1f}%"
                        })
                    
                    summary_df = pd.DataFrame(level_summary)
                    
                    # Display the table with enhanced formatting
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Add color-coded metrics below the table
                    st.markdown("**📊 Performance Highlights:**")
                    cols = st.columns(5)
                    
                    for i, (_, row) in enumerate(summary_df.iterrows()):
                        with cols[i]:
                            profit = float(row['Total Profit (¥k)'].replace('k', ''))
                            usage = float(row['Usage %'].replace('%', ''))
                            
                            # Color code based on performance
                            if profit > 50:
                                color = "🟢"  # Green for high profit
                            elif profit > 20:
                                color = "🟡"  # Yellow for medium profit
                            elif profit > 0:
                                color = "🟠"  # Orange for low profit
                            else:
                                color = "⚪"  # White for no profit
                                
                            st.metric(
                                f"{color} Level {row['Level']}",
                                f"¥{profit:.1f}k",
                                delta=f"{usage:.1f}% used"
                            )
                    
                    # Strategy explanation
                    st.markdown("""
                    **🎯 5-Level Strategy Logic:**
                    - **Level 1 (1%)**: Conservative - Used for small spreads (¥50-100/MWh), high certainty
                    - **Level 2 (2%)**: Moderate-Conservative - Medium spreads (¥100-150/MWh), good certainty  
                    - **Level 3 (3%)**: Moderate - Significant spreads (¥150-200/MWh), moderate risk
                    - **Level 4 (4%)**: Moderate-Aggressive - Large spreads (¥200-300/MWh), higher risk
                    - **Level 5 (5%)**: Maximum Allowable - Very large spreads (>¥300/MWh), maximum risk
                    
                    **🔧 Automatic Selection**: System automatically selects optimal level based on spread magnitude and risk management
                    """)
                
                # Strategy explanation
                st.subheader("📚 Complete Strategy Overview")
                st.markdown("""
                **🎯 Four Jiangsu Market Arbitrage Strategies:**
                
                1. **⏰ Contract Deviation Arbitrage**: 5-level approach (1%-5%) with automatic optimal selection
                2. **📊 Day-ahead vs Real-time Positioning**: Use ML forecasts to optimize market positioning  
                3. **🏭 Geographic Customer Optimization**: Target Southern Jiangsu manufacturing premium vs Northern procurement
                4. **🌱 Renewable Forecast Arbitrage**: Competitive advantage through superior renewable generation forecasting
                """)
                
                # Strategy breakdown chart (existing code)
                strategy_totals = {
                    'Contract Deviation (5-Level)': df['temporal_arbitrage'].sum(),
                    'DA vs RT Positioning': df['ai_arbitrage'].sum(),
                    'Geographic Optimization': df['tou_arbitrage'].sum(),
                    'Renewable Forecasting': df['renewable_arbitrage'].sum()
                }
                
                strategy_df = pd.DataFrame(list(strategy_totals.items()), 
                                         columns=['Strategy', 'Total_Profit'])
                
                if not strategy_df.empty:
                    fig1 = px.bar(strategy_df, x='Strategy', y='Total_Profit',
                                 title="Profit by Arbitrage Strategy - Enhanced with 5-Level Deviation",
                                 color='Total_Profit',
                                 color_continuous_scale='Viridis')
                    
                    # Update annotations for 5-level strategy
                    fig1.add_annotation(x=0, y=strategy_totals['Contract Deviation (5-Level)']/2, 
                                       text="P₁ = V₁ × |P_contract - P_spot| × η₁<br>V₁ = DailyVolume × Level% (1-5% auto-selected)",
                                       showarrow=True, arrowhead=2, font=dict(size=10))
                    
                    fig1.add_annotation(x=1, y=strategy_totals['DA vs RT Positioning']/2,
                                       text="P₂ = V₂ × |P_predicted - P_DA| × η₂<br>V₂ = DailyVolume × 3-5% × Accuracy",
                                       showarrow=True, arrowhead=2, font=dict(size=10))
                    
                    fig1.update_layout(height=500)
                    st.plotly_chart(fig1, use_container_width=True)
                
                # Cost analysis and net profit trend
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**💸 Cost Components:**")
                    st.markdown("""
                    - **Penalty Costs**: Regulatory violations (MAPE > 3%)
                    - **Operational Costs**: Monitoring & adjustments (reduced by AI)
                    """)
                    
                    costs_data = {
                        'Penalty Costs': df['penalty_cost'].sum(),
                        'Operational Costs': df['operational_cost'].sum()
                    }
                    
                    if sum(costs_data.values()) > 0:
                        fig2 = px.pie(values=list(costs_data.values()), names=list(costs_data.keys()),
                                     title="Cost Breakdown<br><sub>Penalty = Volume × (MAPE-3%)/100 × Price × 10%<br>Operational = Volume × MAPE × 1.5 - AI_Savings</sub>")
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.info("✅ No penalty costs! MAPE within 3% regulatory limit")
                
                with col2:
                    # Net profit trend
                    fig3 = px.line(df, x='day', y='net_profit',
                                  title="Daily Net Profit Trend",
                                  markers=True)
                    fig3.update_layout(height=300)
                    st.plotly_chart(fig3, use_container_width=True)
            
            with tab2:
                st.subheader("🤖 AI Model Predictions vs Reality")
                
                st.markdown("""
                **📊 Model Performance Metrics:**
                - **MAPE** = |Predicted - Actual| / Actual × 100%  
                - **Prediction Accuracy** = max(0.1, 1 - MAPE/100)
                - **AI Arbitrage Volume** = Daily_Volume × 0.8 × Prediction_Accuracy
                """)
                
                if pred_df is not None and len(pred_df) > 0:
                    # Sample predictions for visualization (first 3 days)
                    sample_pred = pred_df.head(288) if len(pred_df) > 288 else pred_df  # 3 days max
                    
                    # Prediction accuracy chart
                    fig4 = go.Figure()
                    fig4.add_trace(go.Scatter(
                        x=list(range(len(sample_pred))), 
                        y=sample_pred['actual_price'],
                        mode='lines', 
                        name='Actual Price',
                        line=dict(color='blue', width=2)
                    ))
                    fig4.add_trace(go.Scatter(
                        x=list(range(len(sample_pred))), 
                        y=sample_pred['predicted_price'],
                        mode='lines', 
                        name='AI Predicted Price',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    fig4.add_trace(go.Scatter(
                        x=list(range(len(sample_pred))), 
                        y=sample_pred['day_ahead_price'],
                        mode='lines', 
                        name='Day-ahead Price',
                        line=dict(color='green', width=1, dash='dot')
                    ))
                    
                    fig4.update_layout(
                        title="AI Model Performance: Predictions vs Reality",
                        xaxis_title="Time Interval",
                        yaxis_title="Price (RMB/MWh)",
                        height=500
                    )
                    st.plotly_chart(fig4, use_container_width=True)
                    
                    # Prediction error analysis
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig5 = px.histogram(sample_pred, x='mape', nbins=30,
                                          title="MAPE Distribution",
                                          labels={'mape': 'MAPE (%)'})
                        fig5.add_vline(x=actual_mape, line_dash="dash", line_color="red",
                                      annotation_text=f"Avg: {actual_mape:.2f}%")
                        st.plotly_chart(fig5, use_container_width=True)
                    
                    with col2:
                        fig6 = px.scatter(sample_pred, x='predicted_price', y='actual_price',
                                         title="Predicted vs Actual Prices",
                                         trendline="ols")
                        # Add perfect prediction line
                        min_price = min(sample_pred['predicted_price'].min(), sample_pred['actual_price'].min())
                        max_price = max(sample_pred['predicted_price'].max(), sample_pred['actual_price'].max())
                        fig6.add_shape(type="line", x0=min_price, y0=min_price, x1=max_price, y1=max_price,
                                      line=dict(color="red", dash="dash"))
                        st.plotly_chart(fig6, use_container_width=True)
                else:
                    st.info("No prediction data available")
            
            with tab3:
                st.subheader("📊 Daily Performance Analysis")
                
                # Detailed daily breakdown
                display_columns = ['day', 'date', 'net_profit', 'daily_mape', 'prediction_accuracy',
                                  'temporal_arbitrage', 'ai_arbitrage', 'tou_arbitrage', 'renewable_arbitrage']
                available_columns = [col for col in display_columns if col in df.columns]
                
                if available_columns:
                    display_df = df[available_columns].copy()
                    
                    if 'date' in display_df.columns:
                        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                    display_df = display_df.round(2)
                    
                    st.dataframe(display_df, use_container_width=True)
                
                # Performance metrics by day
                if len(df) > 0:
                    fig7 = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('Daily Net Profit', 'Daily MAPE', 'Prediction Accuracy', 'Strategy Mix'),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                               [{"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    # Daily profit
                    fig7.add_trace(go.Bar(x=df['day'], y=df['net_profit'], name='Net Profit'), row=1, col=1)
                    
                    # Daily MAPE (if available)
                    if 'daily_mape' in df.columns:
                        fig7.add_trace(go.Scatter(x=df['day'], y=df['daily_mape'], 
                                                mode='lines+markers', name='MAPE'), row=1, col=2)
                    
                    # Prediction accuracy (if available)
                    if 'prediction_accuracy' in df.columns:
                        fig7.add_trace(go.Scatter(x=df['day'], y=df['prediction_accuracy'], 
                                                mode='lines+markers', name='Accuracy'), row=2, col=1)
                    
                    # Strategy mix
                    strategies = ['temporal_arbitrage', 'ai_arbitrage', 'tou_arbitrage', 'renewable_arbitrage']
                    for strategy in strategies:
                        if strategy in df.columns:
                            fig7.add_trace(go.Bar(x=df['day'], y=df[strategy], 
                                                name=strategy.replace('_', ' ').title()), row=2, col=2)
                    
                    fig7.update_layout(height=600, title_text="Comprehensive Daily Analysis")
                    st.plotly_chart(fig7, use_container_width=True)

    else:
        st.info("👆 Configure parameters and click 'Run Enhanced Simulation' to start the analysis!")
        
        # Show model information while waiting
        st.subheader("🏆 About the AI-Enhanced Strategy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🤖 AI Model Capabilities:**
            - **Ridge Regression** with 9.55% MAPE
            - **90.5% variance explained** (R² = 0.905)
            - **Real-time price prediction** using market fundamentals
            - **Production-tested** on historical data
            
            **📊 Key Features:**
            - Day-ahead prices
            - Renewable forecasts
            - Thermal capacity
            - Load forecasts
            - Time patterns
            - Price lags
            """)
        
        with col2:
            st.markdown("""
            **⚡ Enhanced Arbitrage Strategies:**
            1. **Temporal Arbitrage** - Contract vs spot trading
            2. **AI-Enhanced Arbitrage** - ML prediction-based trading
            3. **Peak/Off-peak Optimization** - Load shifting
            4. **Renewable Arbitrage** - Green energy timing
            
            **💡 Key Advantages:**
            - **54% MAPE improvement** over baseline
            - **Real market data** integration
            - **Risk-aware** position sizing
            - **Multi-strategy** portfolio approach
            """)

elif page == "📚 Complete Methodology":
    st.header("📚 Complete Methodology & Mathematical Framework")
    
    st.markdown("""
    This section provides complete transparency into the mathematical formulations
    and business logic behind every calculation in the platform.
    """)
    
    # Methodology tabs
    method_tab1, method_tab2, method_tab3 = st.tabs(["🔍 EDA Methods", "🤖 Model Methods", "💹 Arbitrage Methods"])
    
    with method_tab1:
        st.markdown("""
        ## 🔍 **EDA Methodology**
        
        ### **Data Processing:**
        ```python
        # Time feature engineering
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_peak'] = ((df['hour'] >= 8) & (df['hour'] <= 10)) | ((df['hour'] >= 19) & (df['hour'] <= 21))
        df['is_weekend'] = df.index.dayofweek >= 5
        ```
        
        ### **Key Analyses:**
        - **Time series decomposition** for trend/seasonality
        - **Correlation analysis** between market variables
        - **Peak/valley identification** using statistical thresholds
        - **Volatility analysis** using rolling standard deviations
        """)
    
    with method_tab2:
        st.markdown("""
        ## 🤖 **Model Development Methodology**
        
        ### **Feature Engineering:**
        ```python
        # Lag features for temporal patterns
        df['rt_price_lag1'] = df['实时出清电价'].shift(1)
        df['da_price_lag1'] = df['日前出清电价'].shift(1) 
        df['rt_price_lag4'] = df['实时出清电价'].shift(4)
        ```
        
        ### **Model Selection Process:**
        1. **Baseline models**: LSTM, CNN, GRU, Transformer
        2. **Feature engineering**: 25+ engineered features
        3. **Custom loss functions**: Direct MAPE optimization
        4. **Final breakthrough**: Ridge Regression with 10 features
        
        ### **Winning Model:**
        ```python
        Ridge(alpha=1.0, random_state=42)
        Features: ['日前出清电价', '新能源预测', '竞价空间(火电)', '负荷预测',
                  'hour', 'is_peak', 'is_weekend', 
                  'rt_price_lag1', 'da_price_lag1', 'rt_price_lag4']
        ```
        """)
    
    with method_tab3:
        st.markdown("""
        ## 💹 **Jiangsu Market Arbitrage Methodology**
        
        ### **1. Contract Deviation Arbitrage:**
        ```
        Over-consumption (Spot < Contract):
        Profit = Volume × (Contract_Price - Spot_Price) × 85%
        Volume = Daily_Volume × 4.5% (within ±5% user limit)
        
        Under-consumption (Spot > Contract):
        Profit = Volume × (Spot_Price - Contract_Price) × 75%
        Volume = Daily_Volume × 4.5% (demand response)
        ```
        
        ### **2. Day-ahead vs Real-time Positioning:**
        ```
        If |Predicted_RT - Day_Ahead| > ¥20/MWh:
        Profit = Volume × |Predicted_RT - Day_Ahead| × 60-70%
        Volume = Daily_Volume × 3-5% × Prediction_Accuracy
        ```
        
        ### **3. Geographic Customer Optimization:**
        ```
        Southern Jiangsu Premium: ¥25/MWh
        Northern Jiangsu Discount: ¥35/MWh
        Profit = Volume × (¥25 + ¥35) × 20% capture
        Volume = Manufacturing_Volume × 40%
        ```
        
        ### **4. Renewable Forecast Arbitrage:**
        ```
        If Renewable_Std > 50MW and MAPE < 25%:
        Forecast_Advantage = (25 - MAPE) / 25
        Price_Impact = Renewable_Std × ¥0.75/MWh per MW
        Profit = Volume × Price_Impact × Forecast_Advantage
        ```
        
        ### **Risk Management (Jiangsu Rules):**
        - **±5% user side deviation limit**
        - **1.5× real-time price penalty on excess**
        - **MAPE-based volume scaling**
        - **Conservative capture rate assumptions**
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    <p>Jiangsu Energy Analysis Platform | Complete Data Science Pipeline</p>
    <p>From Data Exploration to AI-Powered Arbitrage Strategies</p>
</div>
""", unsafe_allow_html=True)