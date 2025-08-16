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
        """Calculate arbitrage profits using real predictions"""
        day_data = predictions_df.iloc[day_start_idx:day_end_idx]
        
        if len(day_data) == 0:
            return self._zero_profits()
        
        # Calculate daily metrics
        avg_actual = day_data['actual_price'].mean()
        avg_predicted = day_data['predicted_price'].mean()
        avg_day_ahead = day_data['day_ahead_price'].mean()
        daily_mape = day_data['mape'].mean()
        
        # 1. TEMPORAL ARBITRAGE (Contract vs Spot)
        # Available deviation after forecast uncertainty
        forecast_buffer = daily_mape
        available_deviation = max(0, self.max_deviation - forecast_buffer)
        
        # Price spread opportunity
        contract_spot_spread = abs(self.base_price - avg_actual)
        temporal_volume = self.daily_volume * available_deviation / 100
        temporal_profit = temporal_volume * contract_spot_spread * 0.7
        
        # 2. AI-ENHANCED PREDICTION ARBITRAGE
        # Use our model predictions to trade strategically
        prediction_accuracy = max(0.1, 1 - daily_mape / 100)
        
        # Trade on prediction vs day-ahead differences
        da_pred_spread = abs(avg_day_ahead - avg_predicted)
        ai_arbitrage_volume = self.daily_volume * 0.8 * prediction_accuracy
        ai_arbitrage_profit = ai_arbitrage_volume * da_pred_spread * 0.6
        
        # 3. PEAK/OFF-PEAK OPTIMIZATION
        peak_data = day_data[day_data['is_peak'] == 1]
        offpeak_data = day_data[day_data['is_peak'] == 0]
        
        if len(peak_data) > 0 and len(offpeak_data) > 0:
            peak_avg = peak_data['actual_price'].mean()
            offpeak_avg = offpeak_data['actual_price'].mean()
            peak_valley_spread = abs(peak_avg - offpeak_avg)
            
            # Our model helps predict optimal shifting
            shift_efficiency = prediction_accuracy * 0.8
            shift_volume = self.daily_volume * 0.3
            tou_profit = shift_volume * peak_valley_spread * shift_efficiency
        else:
            tou_profit = 0
        
        # 4. RENEWABLE ARBITRAGE
        # Trade based on renewable generation forecasts
        renewable_impact = day_data['renewable_forecast'].std()
        if renewable_impact > 0:
            renewable_volume = self.daily_volume * 0.4
            renewable_spread = renewable_impact * 0.02  # 2% price impact per MW std
            renewable_profit = renewable_volume * renewable_spread * prediction_accuracy
        else:
            renewable_profit = 0
        
        # 5. PENALTY COSTS
        penalty_cost = 0
        if daily_mape > self.max_deviation:
            excess_error = daily_mape - self.max_deviation
            penalty_volume = self.daily_volume * excess_error / 100
            penalty_cost = penalty_volume * self.base_price * self.penalty_rate
        
        # 6. OPERATIONAL COSTS
        # Reduced costs due to better predictions
        base_operational_cost = self.daily_volume * daily_mape * 1.5
        ai_cost_reduction = base_operational_cost * 0.3 * prediction_accuracy  # 30% reduction with good predictions
        operational_cost = base_operational_cost - ai_cost_reduction
        
        # TOTALS
        total_profit = temporal_profit + ai_arbitrage_profit + tou_profit + renewable_profit
        total_costs = penalty_cost + operational_cost
        net_profit = total_profit - total_costs
        
        return {
            'temporal_arbitrage': temporal_profit,
            'ai_arbitrage': ai_arbitrage_profit,
            'tou_arbitrage': tou_profit,
            'renewable_arbitrage': renewable_profit,
            'total_profit': total_profit,
            'penalty_cost': penalty_cost,
            'operational_cost': operational_cost,
            'total_costs': total_costs,
            'net_profit': net_profit,
            'available_deviation': available_deviation,
            'daily_mape': daily_mape,
            'prediction_accuracy': prediction_accuracy,
            'avg_actual_price': avg_actual,
            'avg_predicted_price': avg_predicted,
            'contract_spot_spread': contract_spot_spread
        }
    
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
                
                # Strategy explanation
                st.markdown("""
                **🎯 Four Enhanced Arbitrage Strategies:**
                
                1. **⏰ Temporal Arbitrage**: Exploit differences between contract and spot prices
                2. **📊 ML-Enhanced Arbitrage**: Use machine learning predictions to trade against day-ahead prices  
                3. **📈 Peak/Off-peak Optimization**: Shift demand between peak and valley periods
                4. **🌱 Renewable Arbitrage**: Trade based on renewable generation variability
                """)
                
                # Strategy breakdown chart
                strategy_totals = {
                    'Temporal Arbitrage': df['temporal_arbitrage'].sum(),
                    'ML-Enhanced Arbitrage': df['ai_arbitrage'].sum(),
                    'Peak/Off-peak Optimization': df['tou_arbitrage'].sum(),
                    'Renewable Arbitrage': df['renewable_arbitrage'].sum()
                }
                
                strategy_df = pd.DataFrame(list(strategy_totals.items()), 
                                         columns=['Strategy', 'Total_Profit'])
                
                if not strategy_df.empty:
                    fig1 = px.bar(strategy_df, x='Strategy', y='Total_Profit',
                                 title="Profit by Arbitrage Strategy (with Mathematical Formulations)",
                                 color='Total_Profit',
                                 color_continuous_scale='Viridis')
                    
                    # Add annotations with key equations
                    fig1.add_annotation(x=0, y=strategy_totals['Temporal Arbitrage']/2, 
                                       text="P₁ = V₁ × |P_contract - P_spot| × η₁<br>V₁ = DailyVolume × (3% - MAPE)/100",
                                       showarrow=True, arrowhead=2, font=dict(size=10))
                    
                    fig1.add_annotation(x=1, y=strategy_totals['ML-Enhanced Arbitrage']/2,
                                       text="P₂ = V₂ × |P_predicted - P_DA| × η₂<br>V₂ = DailyVolume × 0.8 × (1-MAPE/100)",
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
        ## 💹 **Arbitrage Strategy Methodology**
        
        ### **1. Temporal Arbitrage:**
        ```
        Profit = Volume × |Contract_Price - Spot_Price| × 70%
        Volume = Daily_Volume × (3% - MAPE) / 100
        ```
        
        ### **2. AI-Enhanced Arbitrage:**
        ```
        Profit = Volume × |AI_Predicted - Day_Ahead| × 60%
        Volume = Daily_Volume × 0.8 × (1 - MAPE/100)
        ```
        
        ### **3. Peak/Off-peak Optimization:**
        ```
        Profit = Shiftable_Volume × Peak_Valley_Spread × Efficiency
        Shiftable_Volume = Daily_Volume × 30%
        Efficiency = Prediction_Accuracy × 80%
        ```
        
        ### **4. Renewable Arbitrage:**
        ```
        Profit = Volume × Price_Impact × Prediction_Accuracy
        Price_Impact = Renewable_Std × 2%
        Volume = Daily_Volume × 40%
        ```
        
        ### **Risk Management:**
        - **3% regulatory deviation limit**
        - **10% penalty rate on violations**
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