import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Energy Price EDA Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load the cleaned energy data"""
    try:
        df = pd.read_csv('cleaned_data/energy_data_cleaned.csv', index_col=0, parse_dates=True)
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please run data_processor.py first.")
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
        st.warning("Original model results not found")
    
    # Improved neural network results
    try:
        improved = pd.read_csv('improved_model_results.csv')
        improved['Phase'] = 'Improved Neural Networks'
        improved = improved.rename(columns={'MAPE_All': 'MAPE'})  # Standardize column name
        results['improved'] = improved
    except FileNotFoundError:
        st.warning("Improved model results not found")
    
    # Final optimization results
    try:
        final = pd.read_csv('final_optimization_results.csv')
        final['Phase'] = 'Final Optimization'
        results['final'] = final
    except FileNotFoundError:
        st.warning("Final optimization results not found")
    
    return results

def create_time_series_plots(df):
    """Create time series plots for price analysis"""
    
    # 1. Price comparison over time
    fig1 = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Day-ahead vs Real-time Prices', 'Price Spread Analysis'),
        vertical_spacing=0.08
    )
    
    # Add price traces
    fig1.add_trace(
        go.Scatter(x=df.index, y=df['日前出清电价'], 
                  name='Day-ahead Price', line=dict(color='blue', width=1)),
        row=1, col=1
    )
    fig1.add_trace(
        go.Scatter(x=df.index, y=df['实时出清电价'], 
                  name='Real-time Price', line=dict(color='red', width=1)),
        row=1, col=1
    )
    
    # Add spread
    spread = df['实时出清电价'] - df['日前出清电价']
    fig1.add_trace(
        go.Scatter(x=df.index, y=spread, 
                  name='Price Spread (RT - DA)', line=dict(color='green', width=1)),
        row=2, col=1
    )
    
    fig1.update_layout(
        height=600,
        title_text="Energy Price Time Series Analysis",
        showlegend=True
    )
    fig1.update_xaxes(title_text="Date", row=2, col=1)
    fig1.update_yaxes(title_text="Price (RMB/MWh)", row=1, col=1)
    fig1.update_yaxes(title_text="Spread (RMB/MWh)", row=2, col=1)
    
    return fig1

def create_daily_patterns(df):
    """Analyze daily and hourly patterns"""
    
    # Create hourly averages
    hourly_stats = df.groupby('hour').agg({
        '日前出清电价': ['mean', 'std'],
        '实时出清电价': ['mean', 'std'],
        '负荷预测': 'mean',
        '新能源预测': 'mean'
    }).round(2)
    
    hourly_stats.columns = ['DA_Price_Mean', 'DA_Price_Std', 'RT_Price_Mean', 'RT_Price_Std', 
                           'Load_Mean', 'Renewable_Mean']
    
    # Create subplots
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Hourly Price Patterns', 'Price Volatility by Hour',
                       'Load vs Renewable Generation', 'Weekend vs Weekday Prices'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": True}, {"secondary_y": False}]]
    )
    
    # Hourly price patterns
    fig2.add_trace(
        go.Scatter(x=hourly_stats.index, y=hourly_stats['DA_Price_Mean'],
                  mode='lines+markers', name='Day-ahead Avg', line=dict(color='blue')),
        row=1, col=1
    )
    fig2.add_trace(
        go.Scatter(x=hourly_stats.index, y=hourly_stats['RT_Price_Mean'],
                  mode='lines+markers', name='Real-time Avg', line=dict(color='red')),
        row=1, col=1
    )
    
    # Price volatility
    fig2.add_trace(
        go.Scatter(x=hourly_stats.index, y=hourly_stats['DA_Price_Std'],
                  mode='lines+markers', name='DA Volatility', line=dict(color='lightblue')),
        row=1, col=2
    )
    fig2.add_trace(
        go.Scatter(x=hourly_stats.index, y=hourly_stats['RT_Price_Std'],
                  mode='lines+markers', name='RT Volatility', line=dict(color='lightcoral')),
        row=1, col=2
    )
    
    # Load vs renewables
    fig2.add_trace(
        go.Scatter(x=hourly_stats.index, y=hourly_stats['Load_Mean'],
                  mode='lines', name='Load', line=dict(color='orange')),
        row=2, col=1
    )
    fig2.add_trace(
        go.Scatter(x=hourly_stats.index, y=hourly_stats['Renewable_Mean'],
                  mode='lines', name='Renewables', line=dict(color='green'), yaxis='y2'),
        row=2, col=1
    )
    
    # Weekend vs weekday comparison
    weekend_prices = df[df['is_weekend'] == 1]['实时出清电价']
    weekday_prices = df[df['is_weekend'] == 0]['实时出清电价']
    
    fig2.add_trace(
        go.Box(y=weekday_prices, name='Weekday', boxpoints='outliers'),
        row=2, col=2
    )
    fig2.add_trace(
        go.Box(y=weekend_prices, name='Weekend', boxpoints='outliers'),
        row=2, col=2
    )
    
    fig2.update_layout(height=800, title_text="Daily and Hourly Pattern Analysis")
    fig2.update_xaxes(title_text="Hour of Day", row=1, col=1)
    fig2.update_xaxes(title_text="Hour of Day", row=1, col=2)
    fig2.update_xaxes(title_text="Hour of Day", row=2, col=1)
    fig2.update_yaxes(title_text="Price (RMB/MWh)", row=1, col=1)
    fig2.update_yaxes(title_text="Std Dev", row=1, col=2)
    fig2.update_yaxes(title_text="Load (MW)", row=2, col=1)
    fig2.update_yaxes(title_text="Renewables (MW)", row=2, col=1, secondary_y=True)
    fig2.update_yaxes(title_text="Price (RMB/MWh)", row=2, col=2)
    
    return fig2, hourly_stats

def create_correlation_analysis(df):
    """Create correlation analysis for price modeling"""
    
    # Select relevant columns for correlation
    price_cols = ['日前出清电价', '实时出清电价']
    feature_cols = ['负荷预测', '外送', '新能源预测', '风电预测', '光伏预测', 
                   '非市场化出力', '竞价空间(火电)', '水电']
    
    corr_data = df[price_cols + feature_cols]
    corr_matrix = corr_data.corr()
    
    # Create correlation heatmap
    fig3 = px.imshow(
        corr_matrix,
        color_continuous_scale='RdBu',
        title="Correlation Matrix: Prices vs Market Factors",
        aspect="auto"
    )
    fig3.update_layout(height=600)
    
    return fig3, corr_matrix

def create_price_distribution_analysis(df):
    """Analyze price distributions and outliers"""
    
    fig4 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Price Distributions', 'Price vs Load Scatter',
                       'Price Spread Distribution', 'Price vs Renewable Generation'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Price distributions
    fig4.add_trace(
        go.Histogram(x=df['日前出清电价'], name='Day-ahead', opacity=0.7, nbinsx=50),
        row=1, col=1
    )
    fig4.add_trace(
        go.Histogram(x=df['实时出清电价'], name='Real-time', opacity=0.7, nbinsx=50),
        row=1, col=1
    )
    
    # Price vs load scatter
    fig4.add_trace(
        go.Scatter(x=df['负荷预测'], y=df['实时出清电价'], 
                  mode='markers', name='Price vs Load', 
                  marker=dict(size=3, opacity=0.6)),
        row=1, col=2
    )
    
    # Price spread distribution
    spread = df['实时出清电价'] - df['日前出清电价']
    fig4.add_trace(
        go.Histogram(x=spread, name='Price Spread', nbinsx=50),
        row=2, col=1
    )
    
    # Price vs renewables
    fig4.add_trace(
        go.Scatter(x=df['新能源预测'], y=df['实时出清电价'], 
                  mode='markers', name='Price vs Renewables',
                  marker=dict(size=3, opacity=0.6, color='green')),
        row=2, col=2
    )
    
    fig4.update_layout(height=800, title_text="Price Distribution and Relationship Analysis")
    fig4.update_xaxes(title_text="Price (RMB/MWh)", row=1, col=1)
    fig4.update_xaxes(title_text="Load Forecast (MW)", row=1, col=2)
    fig4.update_xaxes(title_text="Price Spread (RMB/MWh)", row=2, col=1)
    fig4.update_xaxes(title_text="Renewable Generation (MW)", row=2, col=2)
    fig4.update_yaxes(title_text="Frequency", row=1, col=1)
    fig4.update_yaxes(title_text="Real-time Price (RMB/MWh)", row=1, col=2)
    fig4.update_yaxes(title_text="Frequency", row=2, col=1)
    fig4.update_yaxes(title_text="Real-time Price (RMB/MWh)", row=2, col=2)
    
    return fig4

def calculate_price_insights(df):
    """Calculate key insights for price modeling"""
    
    insights = {}
    
    # Basic statistics
    insights['da_price_mean'] = df['日前出清电价'].mean()
    insights['rt_price_mean'] = df['实时出清电价'].mean()
    insights['price_spread_mean'] = (df['实时出清电价'] - df['日前出清电价']).mean()
    insights['price_volatility_da'] = df['日前出清电价'].std()
    insights['price_volatility_rt'] = df['实时出清电价'].std()
    
    # Peak vs off-peak analysis
    peak_rt_price = df[df['is_peak'] == 1]['实时出清电价'].mean()
    offpeak_rt_price = df[df['is_peak'] == 0]['实时出清电价'].mean()
    insights['peak_premium'] = peak_rt_price - offpeak_rt_price
    
    # Weekend vs weekday
    weekend_rt_price = df[df['is_weekend'] == 1]['实时出清电价'].mean()
    weekday_rt_price = df[df['is_weekend'] == 0]['实时出清电价'].mean()
    insights['weekend_discount'] = weekday_rt_price - weekend_rt_price
    
    # Correlation insights
    load_price_corr = df['负荷预测'].corr(df['实时出清电价'])
    renewable_price_corr = df['新能源预测'].corr(df['实时出清电价'])
    insights['load_price_correlation'] = load_price_corr
    insights['renewable_price_correlation'] = renewable_price_corr
    
    # Price prediction complexity indicators
    da_rt_correlation = df['日前出清电价'].corr(df['实时出清电价'])
    insights['da_rt_correlation'] = da_rt_correlation
    
    # Identify high volatility periods
    hourly_volatility = df.groupby('hour')['实时出清电价'].std()
    insights['most_volatile_hour'] = hourly_volatility.idxmax()
    insights['least_volatile_hour'] = hourly_volatility.idxmin()
    
    return insights

def create_model_comparison_charts(results):
    """Create comprehensive model comparison visualizations"""
    
    # Combine all results
    all_results = []
    colors = {'Original Neural Networks': '#1f77b4', 'Improved Neural Networks': '#ff7f0e', 'Final Optimization': '#2ca02c'}
    
    for phase, df in results.items():
        if df is not None and not df.empty:
            df_copy = df.copy()
            df_copy['Phase'] = df_copy['Phase'].iloc[0] if 'Phase' in df_copy.columns else phase
            all_results.append(df_copy)
    
    if not all_results:
        st.error("No model results found!")
        return None
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # 1. MAPE Comparison (Main Chart)
    fig1 = px.bar(
        combined_df, 
        x='Model', 
        y='MAPE', 
        color='Phase',
        title="🎯 MAPE Comparison Across All Models",
        color_discrete_map=colors,
        text='MAPE'
    )
    
    fig1.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig1.update_layout(
        yaxis_title="MAPE (%)",
        xaxis_title="Model",
        height=500,
        showlegend=True
    )
    
    # Add horizontal line for target MAPE
    fig1.add_hline(y=15, line_dash="dash", line_color="red", 
                   annotation_text="Good Threshold (15%)")
    fig1.add_hline(y=10, line_dash="dash", line_color="green", 
                   annotation_text="Excellent Threshold (10%)")
    
    return fig1, combined_df

def create_performance_metrics_dashboard(combined_df):
    """Create comprehensive performance metrics dashboard"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('RMSE Comparison', 'R² Comparison', 'MAE Comparison', 'Training Efficiency'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": True}]]
    )
    
    # Color mapping
    colors = {'Original Neural Networks': '#1f77b4', 'Improved Neural Networks': '#ff7f0e', 'Final Optimization': '#2ca02c'}
    
    # RMSE comparison
    for phase in combined_df['Phase'].unique():
        phase_data = combined_df[combined_df['Phase'] == phase]
        fig.add_trace(
            go.Bar(x=phase_data['Model'], y=phase_data['RMSE'], 
                   name=f'RMSE - {phase}', marker_color=colors.get(phase, '#333333'),
                   showlegend=False),
            row=1, col=1
        )
    
    # R² comparison
    for phase in combined_df['Phase'].unique():
        phase_data = combined_df[combined_df['Phase'] == phase]
        fig.add_trace(
            go.Bar(x=phase_data['Model'], y=phase_data['R²'], 
                   name=f'R² - {phase}', marker_color=colors.get(phase, '#333333'),
                   showlegend=False),
            row=1, col=2
        )
    
    # MAE comparison
    for phase in combined_df['Phase'].unique():
        phase_data = combined_df[combined_df['Phase'] == phase]
        fig.add_trace(
            go.Bar(x=phase_data['Model'], y=phase_data['MAE'], 
                   name=f'MAE - {phase}', marker_color=colors.get(phase, '#333333'),
                   showlegend=False),
            row=2, col=1
        )
    
    # Training time (if available)
    if 'Training_Time' in combined_df.columns:
        for phase in combined_df['Phase'].unique():
            phase_data = combined_df[combined_df['Phase'] == phase]
            if not phase_data['Training_Time'].isna().all():
                fig.add_trace(
                    go.Bar(x=phase_data['Model'], y=phase_data['Training_Time'], 
                           name=f'Time - {phase}', marker_color=colors.get(phase, '#333333'),
                           showlegend=False),
                    row=2, col=2
                )
    
    fig.update_layout(height=800, title_text="📊 Comprehensive Model Performance Metrics")
    fig.update_xaxes(title_text="Model", row=2, col=1)
    fig.update_xaxes(title_text="Model", row=2, col=2)
    fig.update_yaxes(title_text="RMSE", row=1, col=1)
    fig.update_yaxes(title_text="R²", row=1, col=2)
    fig.update_yaxes(title_text="MAE", row=2, col=1)
    fig.update_yaxes(title_text="Training Time (s)", row=2, col=2)
    
    return fig

def create_model_evolution_chart(combined_df):
    """Show the evolution of model performance across phases"""
    
    # Calculate phase averages
    phase_summary = combined_df.groupby('Phase').agg({
        'MAPE': ['mean', 'min'],
        'RMSE': ['mean', 'min'], 
        'R²': ['mean', 'max']
    }).round(3)
    
    phase_summary.columns = ['MAPE_Mean', 'MAPE_Best', 'RMSE_Mean', 'RMSE_Best', 'R²_Mean', 'R²_Best']
    phase_summary = phase_summary.reset_index()
    
    # Create evolution chart
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('MAPE Evolution (Lower is Better)', 'R² Evolution (Higher is Better)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # MAPE evolution
    fig.add_trace(
        go.Scatter(x=phase_summary['Phase'], y=phase_summary['MAPE_Best'],
                  mode='lines+markers', name='Best MAPE', 
                  line=dict(color='green', width=3), marker=dict(size=10)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=phase_summary['Phase'], y=phase_summary['MAPE_Mean'],
                  mode='lines+markers', name='Average MAPE', 
                  line=dict(color='lightblue', width=2, dash='dash')),
        row=1, col=1
    )
    
    # R² evolution
    fig.add_trace(
        go.Scatter(x=phase_summary['Phase'], y=phase_summary['R²_Best'],
                  mode='lines+markers', name='Best R²', 
                  line=dict(color='blue', width=3), marker=dict(size=10)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=phase_summary['Phase'], y=phase_summary['R²_Mean'],
                  mode='lines+markers', name='Average R²', 
                  line=dict(color='lightcoral', width=2, dash='dash')),
        row=1, col=2
    )
    
    fig.update_layout(height=500, title_text="📈 Model Performance Evolution Across Phases")
    fig.update_xaxes(title_text="Development Phase", row=1, col=1)
    fig.update_xaxes(title_text="Development Phase", row=1, col=2)
    fig.update_yaxes(title_text="MAPE (%)", row=1, col=1)
    fig.update_yaxes(title_text="R²", row=1, col=2)
    
    return fig, phase_summary

def show_model_comparison_page():
    """Display the model comparison page"""
    st.title("🤖 Model Comparison Dashboard")
    st.markdown("### Complete Energy Price Prediction Model Performance Analysis")
    
    # Load model results
    results = load_model_results()
    
    # Create main comparison chart
    main_chart, combined_df = create_model_comparison_charts(results)
    
    if main_chart is None:
        return
    
    # Show main MAPE comparison
    st.plotly_chart(main_chart, use_container_width=True)
    
    # Winner announcement
    best_model = combined_df.loc[combined_df['MAPE'].idxmin()]
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🏆 Best Model", 
            best_model['Model'],
            delta=f"{best_model['Phase']}"
        )
    
    with col2:
        st.metric(
            "🎯 Best MAPE", 
            f"{best_model['MAPE']:.2f}%",
            delta="Excellent!" if best_model['MAPE'] < 12 else "Good" if best_model['MAPE'] < 18 else "Fair"
        )
    
    with col3:
        st.metric(
            "📊 R² Score", 
            f"{best_model['R²']:.3f}",
            delta=f"Explains {best_model['R²']*100:.1f}% of variance"
        )
    
    # Performance status
    st.markdown("---")
    
    if best_model['MAPE'] < 12:
        st.success("✅ **EXCELLENT PERFORMANCE** - Ready for production deployment!")
    elif best_model['MAPE'] < 18:
        st.success("✅ **GOOD PERFORMANCE** - Suitable for most trading applications")
    elif best_model['MAPE'] < 25:
        st.warning("⚠️ **FAIR PERFORMANCE** - Use with caution, consider ensemble methods")
    else:
        st.error("❌ **POOR PERFORMANCE** - Requires different modeling approach")
    
    # Detailed performance metrics
    st.subheader("📊 Detailed Performance Metrics")
    metrics_chart = create_performance_metrics_dashboard(combined_df)
    st.plotly_chart(metrics_chart, use_container_width=True)
    
    # Model evolution
    st.subheader("📈 Model Development Evolution")
    evolution_chart, phase_summary = create_model_evolution_chart(combined_df)
    st.plotly_chart(evolution_chart, use_container_width=True)
    
    # Phase summary table
    st.subheader("📋 Phase Summary Statistics")
    st.dataframe(phase_summary.round(3), use_container_width=True)
    
    # Complete results table
    st.subheader("🔢 Complete Model Results")
    
    # Format the display dataframe
    display_df = combined_df[['Model', 'Phase', 'MAPE', 'RMSE', 'R²', 'MAE']].copy()
    display_df = display_df.sort_values('MAPE')
    
    # Highlight the best model
    def highlight_best(row):
        if row['MAPE'] == display_df['MAPE'].min():
            return ['background-color: gold'] * len(row)
        else:
            return [''] * len(row)
    
    styled_df = display_df.style.apply(highlight_best, axis=1).format({
        'MAPE': '{:.2f}%',
        'RMSE': '{:.2f}',
        'R²': '{:.3f}',
        'MAE': '{:.2f}'
    })
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Key insights
    st.subheader("💡 Key Insights & Recommendations")
    
    mape_improvement = 20.89 - best_model['MAPE']  # From original baseline
    improvement_pct = mape_improvement / 20.89 * 100
    
    insights = f"""
    **🎯 Performance Breakthrough:**
    - Achieved **{mape_improvement:.2f} percentage point** MAPE reduction ({improvement_pct:.1f}% relative improvement)
    - Final MAPE of **{best_model['MAPE']:.2f}%** vs original baseline of **20.89%**
    - Model explains **{best_model['R²']*100:.1f}%** of price variance
    
    **🔍 Key Discoveries:**
    - **Simplicity beats complexity**: Linear models outperformed deep neural networks
    - **Feature selection critical**: 10 features performed better than 25 features  
    - **Traditional ML wins**: Ridge Regression achieved best results
    - **MSE loss optimal**: Custom MAPE loss degraded performance
    
    **🚀 Production Readiness:**
    - **{best_model['Model']}** is recommended for deployment
    - Excellent accuracy for energy trading applications
    - Fast inference and high interpretability
    - Ready for integration with arbitrage simulator
    """
    
    st.markdown(insights)
    
    # Download results
    st.subheader("📥 Download Results")
    csv = combined_df.to_csv(index=False)
    st.download_button(
        label="Download Complete Results CSV",
        data=csv,
        file_name="complete_model_comparison_results.csv",
        mime="text/csv"
    )

def main():
    st.title("⚡ Energy Price Exploration Dashboard")
    st.markdown("### Comprehensive EDA for Price Prediction Model Development")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Page:",
        ["📊 EDA Analysis", "🤖 Model Comparison"]
    )
    
    if page == "📊 EDA Analysis":
        analysis_type = st.sidebar.selectbox(
            "Choose Analysis Type:",
            ["Overview & Summary", "Time Series Analysis", "Daily Patterns", 
             "Correlation Analysis", "Distribution Analysis", "Price Insights"]
        )
    else:
        analysis_type = None
    
    # Route to appropriate page
    if page == "🤖 Model Comparison":
        show_model_comparison_page()
        return
    
    # EDA Analysis content
    if analysis_type == "Overview & Summary":
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Time Period", "July 2025")
        with col3:
            st.metric("Frequency", "15-min intervals")
        with col4:
            st.metric("Features", len(df.columns))
        
        st.subheader("Sample Data")
        st.dataframe(df.head(10))
        
        st.subheader("Basic Statistics")
        st.dataframe(df.describe())
        
    elif analysis_type == "Time Series Analysis":
        st.header("📈 Time Series Analysis")
        fig1 = create_time_series_plots(df)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("""
        **Key Observations:**
        - Compare day-ahead vs real-time price patterns
        - Analyze price spread volatility
        - Identify temporal trends and anomalies
        """)
        
    elif analysis_type == "Daily Patterns":
        st.header("🕐 Daily & Hourly Patterns")
        fig2, hourly_stats = create_daily_patterns(df)
        st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Hourly Statistics Summary")
        st.dataframe(hourly_stats)
        
    elif analysis_type == "Correlation Analysis":
        st.header("🔗 Correlation Analysis")
        fig3, corr_matrix = create_correlation_analysis(df)
        st.plotly_chart(fig3, use_container_width=True)
        
        st.subheader("Top Correlations with Real-time Price")
        rt_corr = corr_matrix['实时出清电价'].abs().sort_values(ascending=False)
        st.dataframe(rt_corr[1:].head(10))  # Exclude self-correlation
        
    elif analysis_type == "Distribution Analysis":
        st.header("📊 Price Distribution Analysis")
        fig4 = create_price_distribution_analysis(df)
        st.plotly_chart(fig4, use_container_width=True)
        
    elif analysis_type == "Price Insights":
        st.header("💡 Price Modeling Insights")
        insights = calculate_price_insights(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Price Statistics")
            st.metric("Avg Day-ahead Price", f"{insights['da_price_mean']:.2f} RMB/MWh")
            st.metric("Avg Real-time Price", f"{insights['rt_price_mean']:.2f} RMB/MWh")
            st.metric("Avg Price Spread", f"{insights['price_spread_mean']:.2f} RMB/MWh")
            st.metric("Peak Hour Premium", f"{insights['peak_premium']:.2f} RMB/MWh")
            
        with col2:
            st.subheader("Modeling Indicators")
            st.metric("Load-Price Correlation", f"{insights['load_price_correlation']:.3f}")
            st.metric("Renewable-Price Correlation", f"{insights['renewable_price_correlation']:.3f}")
            st.metric("DA-RT Price Correlation", f"{insights['da_rt_correlation']:.3f}")
            st.metric("Weekend Discount", f"{insights['weekend_discount']:.2f} RMB/MWh")
        
        st.subheader("📋 Modeling Recommendations")
        
        recommendations = []
        
        if insights['da_rt_correlation'] > 0.7:
            recommendations.append("✅ **Strong DA-RT correlation** - Day-ahead prices are good predictors")
        else:
            recommendations.append("⚠️ **Weak DA-RT correlation** - Need additional features beyond day-ahead prices")
            
        if abs(insights['load_price_correlation']) > 0.5:
            recommendations.append("✅ **Significant load correlation** - Include demand forecasts as key features")
        else:
            recommendations.append("⚠️ **Weak load correlation** - Price driven by other market factors")
            
        if abs(insights['renewable_price_correlation']) > 0.3:
            recommendations.append("✅ **Renewable impact detected** - Include renewable generation forecasts")
            
        if insights['peak_premium'] > 50:
            recommendations.append("✅ **Strong peak pricing** - Time-of-day features critical")
            
        if insights['price_volatility_rt'] > 100:
            recommendations.append("⚠️ **High price volatility** - Consider volatility modeling approaches")
            
        for rec in recommendations:
            st.markdown(rec)
        
        st.subheader("🎯 Next Steps for Model Development")
        st.markdown("""
        1. **Feature Engineering**: Create lag features, moving averages, and interaction terms
        2. **Model Selection**: Consider time series models (ARIMA, LSTM) vs regression approaches
        3. **Validation Strategy**: Use time-based cross-validation for temporal data
        4. **Target Variable**: Focus on real-time price prediction as primary objective
        5. **Evaluation Metrics**: Use MAPE, RMSE considering business impact of forecast errors
        """)

if __name__ == "__main__":
    main()