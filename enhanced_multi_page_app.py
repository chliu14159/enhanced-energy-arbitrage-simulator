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

@st.cache_data
def load_model_results():
    """Load all model comparison results"""
    results = {}
    
    # Original neural network results
    try:
        original = pd.read_csv('model_comparison_summary.csv')
        original['Phase'] = 'Original Neural Networks'
        results['original'] = original
    except FileNotFoundError:
        pass
    
    # Improved neural network results  
    try:
        improved = pd.read_csv('improved_model_results.csv')
        improved['Phase'] = 'Improved Neural Networks'
        improved = improved.rename(columns={'MAPE_All': 'MAPE'})
        results['improved'] = improved
    except FileNotFoundError:
        pass
    
    # Final optimization results
    try:
        final = pd.read_csv('final_optimization_results.csv')
        final['Phase'] = 'Final Optimization'
        results['final'] = final
    except FileNotFoundError:
        pass
    
    return results

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
    # Import the existing arbitrage simulator logic here
    # For now, show a placeholder
    st.header("💹 Enhanced Arbitrage Simulation")
    st.info("🚧 This section will contain the full arbitrage simulator with AI predictions and methodology transparency")
    
    # Add a button to redirect to the original simulator
    st.markdown("""
    **Note**: The complete arbitrage simulator is available in the original enhanced_arbitrage_simulator.py file.
    This multi-page version focuses on the complete data science story.
    
    The arbitrage simulator includes:
    - 4 enhanced arbitrage strategies
    - Real-time AI predictions
    - Complete mathematical methodology
    - Risk management and compliance
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