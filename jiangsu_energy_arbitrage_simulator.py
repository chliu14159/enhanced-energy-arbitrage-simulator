import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Jiangsu Energy Arbitrage Simulator",
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
.sidebar .sidebar-content {
    background-color: #f8f9fa;
}
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<div class="main-header"><h1>⚡ Jiangsu Energy Market Arbitrage Simulator</h1><p>3-Day Dynamic Simulation with Risk Management</p></div>', 
            unsafe_allow_html=True)

# Initialize session state for simulation data
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = None
if 'current_day' not in st.session_state:
    st.session_state.current_day = 1

class EnergyArbitrageSimulator:
    def __init__(self, portfolio_size, base_price, volatility):
        self.portfolio_size = portfolio_size  # GWh/year
        self.base_price = base_price  # RMB/MWh
        self.volatility = volatility  # Price volatility factor
        self.daily_volume = portfolio_size * 1000 / 365  # MWh/day
        self.monthly_volume = portfolio_size * 1000 / 12  # MWh/month
        
        # Risk parameters
        self.max_deviation = 3.0  # Regulatory limit %
        self.penalty_rate = 0.1  # 10% penalty on excess deviation
        self.confidence_level = 0.95  # VaR confidence level
        
    def generate_market_conditions(self, day):
        """Generate realistic market conditions for 3-day simulation"""
        np.random.seed(42 + day)  # Reproducible but different per day
        
        # Base patterns - different for each day
        day_factors = {
            0: {"temp": 32, "demand_factor": 1.0, "wind": 500, "solar": 300},  # t-1
            1: {"temp": 35, "demand_factor": 1.15, "wind": 350, "solar": 250},  # t (hot day)
            2: {"temp": 28, "demand_factor": 0.9, "wind": 650, "solar": 400}   # t+1 (cooler)
        }
        
        base_conditions = day_factors[day]
        
        # Price generation with realistic patterns
        base_price_adj = self.base_price * base_conditions["demand_factor"]
        
        # 96-point pricing (15-min intervals)
        hourly_pattern = np.array([
            0.7, 0.6, 0.5, 0.5, 0.6, 0.8,  # 00-06: Valley
            1.0, 1.3, 1.5, 1.4, 1.2, 1.1,  # 06-12: Morning peak
            1.0, 0.9, 1.0, 1.1, 1.2, 1.4,  # 12-18: Afternoon
            1.6, 1.8, 1.5, 1.2, 1.0, 0.8   # 18-24: Evening peak
        ])
        
        prices_96 = np.repeat(hourly_pattern, 4) * base_price_adj
        prices_96 += np.random.normal(0, base_price_adj * self.volatility * 0.3, 96)
        
        # Market data
        market_data = {
            'day': day,
            'date': datetime(2024, 7, 15) + timedelta(days=day),
            'temperature': base_conditions["temp"] + np.random.normal(0, 1),
            'wind_forecast': base_conditions["wind"] + np.random.normal(0, 50),
            'solar_forecast': base_conditions["solar"] + np.random.normal(0, 30),
            'demand_factor': base_conditions["demand_factor"],
            'contract_price': self.base_price,
            'day_ahead_price': np.mean(prices_96),
            'real_time_avg': np.mean(prices_96) + np.random.normal(0, 20),
            'south_jiangsu_price': np.mean(prices_96) + 25,
            'north_jiangsu_price': np.mean(prices_96) - 15,
            'prices_96': prices_96,
            'peak_price': np.percentile(prices_96, 90),
            'valley_price': np.percentile(prices_96, 10)
        }
        
        return market_data
    
    def simulate_forecast_accuracy(self, market_data, target_mape):
        """Simulate load forecasting with specified MAPE"""
        # True customer load based on market conditions
        base_load = self.daily_volume * market_data['demand_factor']
        
        # Temperature sensitivity (3% per degree above 30°C for commercial loads)
        if market_data['temperature'] > 30:
            temp_adjustment = (market_data['temperature'] - 30) * 0.03
            true_load = base_load * (1 + temp_adjustment)
        else:
            true_load = base_load
        
        # Add random variation
        true_load *= (1 + np.random.normal(0, 0.05))  # 5% base variation
        
        # Generate forecast with target MAPE
        forecast_error = np.random.normal(0, target_mape/100 * 0.4)  # MAPE to std conversion
        forecasted_load = true_load * (1 + forecast_error)
        
        # Ensure realistic bounds
        forecasted_load = max(0, forecasted_load)
        true_load = max(0, true_load)
        
        actual_mape = abs(forecasted_load - true_load) / true_load * 100
        
        return {
            'forecasted_load': forecasted_load,
            'actual_load': true_load,
            'forecast_error': forecasted_load - true_load,
            'actual_mape': actual_mape,
            'error_pct': forecast_error * 100
        }
    
    def calculate_arbitrage_profits(self, market_data, load_data):
        """Calculate profits from different arbitrage strategies"""
        
        # 1. TEMPORAL ARBITRAGE
        forecast_buffer = load_data['actual_mape']
        available_deviation = max(0, self.max_deviation - forecast_buffer)
        
        # Contract vs spot opportunity
        price_spread = abs(market_data['contract_price'] - market_data['real_time_avg'])
        temporal_volume = self.daily_volume * available_deviation / 100
        temporal_profit = temporal_volume * price_spread * 0.7  # 70% capture efficiency
        
        # 2. ZONAL ARBITRAGE
        zonal_spread = market_data['south_jiangsu_price'] - market_data['north_jiangsu_price']
        zonal_volume = self.daily_volume * 0.6  # 60% of volume
        # Reduced by forecast error (scheduling accuracy)
        zonal_efficiency = max(0.3, 1 - load_data['actual_mape'] / 100)
        zonal_profit = zonal_volume * zonal_spread * zonal_efficiency
        
        # 3. TIME-OF-USE ARBITRAGE
        peak_valley_spread = market_data['peak_price'] - market_data['valley_price']
        dr_capacity = self.daily_volume * 0.4  # 40% flexible load
        # Success rate depends on forecast accuracy
        peak_prediction_accuracy = max(0.2, 1 - load_data['actual_mape'] / 50)
        tou_profit = dr_capacity * peak_valley_spread * peak_prediction_accuracy * 0.25  # 25% of day in peaks
        
        # 4. PENALTY COSTS
        penalty_cost = 0
        if load_data['actual_mape'] > self.max_deviation:
            excess_error = load_data['actual_mape'] - self.max_deviation
            penalty_volume = self.daily_volume * excess_error / 100
            penalty_cost = penalty_volume * self.base_price * self.penalty_rate
        
        # 5. OPERATIONAL COSTS (hedging, monitoring, corrections)
        operational_cost = self.daily_volume * load_data['actual_mape'] * 1.5  # RMB 1.5/MWh per % MAPE
        
        total_profit = temporal_profit + zonal_profit + tou_profit
        total_costs = penalty_cost + operational_cost
        net_profit = total_profit - total_costs
        
        return {
            'temporal_arbitrage': temporal_profit,
            'zonal_arbitrage': zonal_profit,
            'tou_arbitrage': tou_profit,
            'total_profit': total_profit,
            'penalty_cost': penalty_cost,
            'operational_cost': operational_cost,
            'total_costs': total_costs,
            'net_profit': net_profit,
            'available_deviation': available_deviation
        }
    
    def calculate_var_risk(self, profits_series, confidence_level=0.95):
        """Calculate Value at Risk and risk metrics"""
        if len(profits_series) < 2:
            return {
                'var_95': 0,
                'var_99': 0,
                'expected_shortfall': 0,
                'volatility': 0,
                'max_drawdown': 0
            }
        
        # Value at Risk
        var_95 = np.percentile(profits_series, (1 - 0.95) * 100)
        var_99 = np.percentile(profits_series, (1 - 0.99) * 100)
        
        # Expected Shortfall (Conditional VaR)
        tail_losses = profits_series[profits_series <= var_95]
        expected_shortfall = np.mean(tail_losses) if len(tail_losses) > 0 else var_95
        
        # Risk metrics
        volatility = np.std(profits_series)
        max_drawdown = np.min(profits_series) - np.max(profits_series)
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall': expected_shortfall,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': np.mean(profits_series) / volatility if volatility > 0 else 0
        }

# Sidebar Controls
with st.sidebar:
    st.header("🎛️ Simulation Parameters")
    
    # Portfolio parameters
    st.subheader("Portfolio Configuration")
    portfolio_size = st.slider("Portfolio Size (GWh/year)", 500, 3000, 1000, 100)
    base_price = st.slider("Base Energy Price (RMB/MWh)", 350, 550, 450, 10)
    volatility = st.slider("Market Volatility Factor", 0.1, 0.5, 0.2, 0.05)
    
    # Forecast accuracy scenarios
    st.subheader("📊 Forecast Accuracy Scenarios")
    mape_scenarios = st.multiselect(
        "MAPE Scenarios to Compare (%)",
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
        default=[1.5, 3.0, 5.0]
    )
    
    # Risk management
    st.subheader("⚠️ Risk Management")
    var_confidence = st.select_slider("VaR Confidence Level", [0.90, 0.95, 0.99], value=0.95)
    max_daily_risk = st.slider("Max Daily Risk Limit (¥k)", 100, 1000, 500, 50)
    
    # Simulation control
    st.subheader("🎮 Simulation Control")
    if st.button("🚀 Run 3-Day Simulation", type="primary"):
        # Initialize simulator
        simulator = EnergyArbitrageSimulator(portfolio_size, base_price, volatility)
        
        # Run simulation for all scenarios and days
        simulation_results = []
        
        for mape in mape_scenarios:
            for day in range(3):  # t-1, t, t+1
                market_data = simulator.generate_market_conditions(day)
                load_data = simulator.simulate_forecast_accuracy(market_data, mape)
                profits = simulator.calculate_arbitrage_profits(market_data, load_data)
                
                simulation_results.append({
                    'day': day,
                    'day_name': ['t-1', 't', 't+1'][day],
                    'mape_scenario': mape,
                    'date': market_data['date'],
                    'temperature': market_data['temperature'],
                    'forecasted_load': load_data['forecasted_load'],
                    'actual_load': load_data['actual_load'],
                    'actual_mape': load_data['actual_mape'],
                    'forecast_error': load_data['forecast_error'],
                    **profits,
                    **market_data
                })
        
        # Store results in session state
        st.session_state.simulation_data = pd.DataFrame(simulation_results)
        st.success("✅ Simulation completed!")

# Main content area
if st.session_state.simulation_data is not None:
    df = st.session_state.simulation_data
    
    # Tab layout for organized presentation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "📊 3-Day Analysis", "⚡ Strategy Performance", 
        "⚠️ Risk Analysis", "💹 Financial Metrics"
    ])
    
    with tab1:
        st.header("📈 Simulation Overview")
        
        # Key metrics summary
        col1, col2, col3, col4 = st.columns(4)
        
        avg_profit = df['net_profit'].mean()
        best_scenario = df.loc[df['net_profit'].idxmax()]
        worst_scenario = df.loc[df['net_profit'].idxmin()]
        total_volume = df['actual_load'].sum()
        
        with col1:
            st.metric("Average Daily Profit", f"¥{avg_profit/1000:.1f}k", 
                     delta=f"{avg_profit/portfolio_size:.2f} RMB/MWh")
        
        with col2:
            st.metric("Best Scenario", 
                     f"¥{best_scenario['net_profit']/1000:.1f}k",
                     delta=f"{best_scenario['mape_scenario']}% MAPE, Day {best_scenario['day_name']}")
        
        with col3:
            st.metric("Worst Scenario", 
                     f"¥{worst_scenario['net_profit']/1000:.1f}k",
                     delta=f"{worst_scenario['mape_scenario']}% MAPE, Day {worst_scenario['day_name']}")
        
        with col4:
            st.metric("Total Volume", f"{total_volume:.0f} MWh",
                     delta=f"{len(mape_scenarios)} scenarios × 3 days")
        
        # Overview visualization
        fig = px.box(df, x='mape_scenario', y='net_profit', color='day_name',
                     title="Profit Distribution by MAPE Scenario and Day",
                     labels={'net_profit': 'Net Profit (RMB)', 'mape_scenario': 'Forecast MAPE (%)'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("📊 3-Day Dynamic Analysis")
        
        # Day-by-day comparison
        selected_mape = st.selectbox("Select MAPE Scenario for Detailed Analysis", mape_scenarios)
        
        day_data = df[df['mape_scenario'] == selected_mape].copy()
        day_data = day_data.sort_values('day')
        
        # Market conditions evolution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=day_data['day_name'], y=day_data['temperature'],
                                    mode='lines+markers', name='Temperature (°C)',
                                    line=dict(color='red', width=3)))
            fig.add_trace(go.Scatter(x=day_data['day_name'], y=day_data['wind_forecast']/10,
                                    mode='lines+markers', name='Wind Forecast (÷10 MW)',
                                    line=dict(color='blue', width=3), yaxis='y2'))
            
            fig.update_layout(
                title=f"Market Conditions Evolution (MAPE {selected_mape}%)",
                yaxis=dict(title="Temperature (°C)"),
                yaxis2=dict(title="Wind (÷10 MW)", overlaying="y", side="right"),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=day_data['day_name'], y=day_data['forecasted_load'],
                                    mode='lines+markers', name='Forecasted Load',
                                    line=dict(color='blue', dash='dash')))
            fig.add_trace(go.Scatter(x=day_data['day_name'], y=day_data['actual_load'],
                                    mode='lines+markers', name='Actual Load',
                                    line=dict(color='green', width=3)))
            
            fig.update_layout(
                title=f"Load Forecast vs Actual (MAPE {selected_mape}%)",
                yaxis_title="Load (MWh)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Profit evolution across days
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Net Profit by Day', 'Strategy Breakdown', 
                           'Forecast Accuracy', 'Available Arbitrage Capacity'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Net profit
        fig.add_trace(go.Bar(x=day_data['day_name'], y=day_data['net_profit']/1000,
                            name='Net Profit', marker_color='green'), row=1, col=1)
        
        # Strategy breakdown
        fig.add_trace(go.Bar(x=day_data['day_name'], y=day_data['temporal_arbitrage']/1000,
                            name='Temporal'), row=1, col=2)
        fig.add_trace(go.Bar(x=day_data['day_name'], y=day_data['zonal_arbitrage']/1000,
                            name='Zonal'), row=1, col=2)
        fig.add_trace(go.Bar(x=day_data['day_name'], y=day_data['tou_arbitrage']/1000,
                            name='TOU'), row=1, col=2)
        
        # Forecast accuracy
        fig.add_trace(go.Scatter(x=day_data['day_name'], y=day_data['actual_mape'],
                                mode='lines+markers', name='Actual MAPE'), row=2, col=1)
        
        # Available capacity
        fig.add_trace(go.Bar(x=day_data['day_name'], y=day_data['available_deviation'],
                            name='Available Dev %', marker_color='orange'), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=True,
                         title_text=f"3-Day Analysis: MAPE {selected_mape}%")
        fig.update_yaxes(title_text="Profit (¥k)", row=1, col=1)
        fig.update_yaxes(title_text="Profit (¥k)", row=1, col=2)
        fig.update_yaxes(title_text="MAPE (%)", row=2, col=1)
        fig.update_yaxes(title_text="Available %", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("⚡ Strategy Performance Analysis")
        
        # Performance by strategy and scenario
        strategy_cols = ['temporal_arbitrage', 'zonal_arbitrage', 'tou_arbitrage']
        
        # Melt data for better visualization
        strategy_data = df.melt(
            id_vars=['mape_scenario', 'day_name'], 
            value_vars=strategy_cols,
            var_name='strategy', 
            value_name='profit'
        )
        
        strategy_data['profit_k'] = strategy_data['profit'] / 1000
        
        fig = px.box(strategy_data, x='mape_scenario', y='profit_k', color='strategy',
                     title="Strategy Performance by MAPE Scenario",
                     labels={'profit_k': 'Profit (¥k)', 'mape_scenario': 'Forecast MAPE (%)'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Strategy correlation matrix
        st.subheader("📊 Strategy Correlation Analysis")
        
        corr_data = df[strategy_cols + ['actual_mape']].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=corr_data.columns,
            y=corr_data.columns,
            colorscale='RdBu',
            zmid=0
        ))
        fig.update_layout(title="Strategy Correlation Matrix", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance sensitivity
        st.subheader("🎯 MAPE Sensitivity Analysis")
        
        sensitivity_stats = df.groupby('mape_scenario').agg({
            'net_profit': ['mean', 'std', 'min', 'max'],
            'temporal_arbitrage': 'mean',
            'zonal_arbitrage': 'mean', 
            'tou_arbitrage': 'mean',
            'penalty_cost': 'mean'
        }).round(0)
        
        # Flatten column names
        sensitivity_stats.columns = ['_'.join(col).strip() for col in sensitivity_stats.columns]
        sensitivity_stats = sensitivity_stats.reset_index()
        
        st.dataframe(sensitivity_stats, use_container_width=True)
    
    with tab4:
        st.header("⚠️ Risk Analysis & VaR")
        
        # Calculate VaR for each MAPE scenario
        simulator = EnergyArbitrageSimulator(portfolio_size, base_price, volatility)
        
        risk_summary = []
        for mape in mape_scenarios:
            scenario_profits = df[df['mape_scenario'] == mape]['net_profit'].values
            risk_metrics = simulator.calculate_var_risk(scenario_profits, var_confidence)
            risk_metrics['mape_scenario'] = mape
            risk_metrics['mean_profit'] = np.mean(scenario_profits)
            risk_summary.append(risk_metrics)
        
        risk_df = pd.DataFrame(risk_summary)
        
        # Risk metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📉 Value at Risk (VaR)")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=risk_df['mape_scenario'], y=risk_df['var_95']/1000,
                                name=f'VaR {var_confidence*100:.0f}%', marker_color='red'))
            fig.update_layout(title=f"VaR {var_confidence*100:.0f}% by MAPE",
                             xaxis_title="MAPE (%)", yaxis_title="VaR (¥k)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Risk-Return Profile")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=risk_df['volatility']/1000, y=risk_df['mean_profit']/1000,
                mode='markers+text', text=risk_df['mape_scenario'],
                textposition="top center",
                marker=dict(size=15, color=risk_df['mape_scenario'], colorscale='RdYlGn_r')
            ))
            fig.update_layout(title="Risk-Return Scatter",
                             xaxis_title="Volatility (¥k)", yaxis_title="Mean Profit (¥k)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.subheader("⚡ Sharpe Ratio")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=risk_df['mape_scenario'], y=risk_df['sharpe_ratio'],
                                name='Sharpe Ratio', marker_color='blue'))
            fig.update_layout(title="Risk-Adjusted Returns",
                             xaxis_title="MAPE (%)", yaxis_title="Sharpe Ratio")
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk limit monitoring
        st.subheader("🚨 Risk Limit Monitoring")
        
        risk_violations = df[abs(df['net_profit']) > max_daily_risk * 1000]
        
        col1, col2 = st.columns(2)
        
        with col1:
            violation_count = len(risk_violations)
            total_scenarios = len(df)
            violation_rate = violation_count / total_scenarios * 100
            
            st.metric("Risk Limit Violations", f"{violation_count}/{total_scenarios}",
                     delta=f"{violation_rate:.1f}% of scenarios")
            
            if violation_count > 0:
                st.warning(f"⚠️ Risk limit exceeded in {violation_count} scenarios")
                st.dataframe(risk_violations[['mape_scenario', 'day_name', 'net_profit']].round(0))
        
        with col2:
            # Risk heatmap
            risk_matrix = df.pivot_table(
                values='net_profit', index='mape_scenario', 
                columns='day_name', aggfunc='mean'
            )
            
            fig = go.Figure(data=go.Heatmap(
                z=risk_matrix.values/1000,
                x=risk_matrix.columns,
                y=risk_matrix.index,
                colorscale='RdYlGn',
                colorbar=dict(title="Profit (¥k)")
            ))
            fig.update_layout(title="Risk Heatmap: Profit by MAPE & Day", height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("💹 Financial Performance Metrics")
        
        # Portfolio-level metrics
        total_annual_profit = df['net_profit'].sum() * 365/3  # Annualize
        portfolio_value = portfolio_size * 1000 * base_price  # Total portfolio value
        roi = (total_annual_profit / portfolio_value) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Annualized Profit", f"¥{total_annual_profit/1e6:.1f}M")
        
        with col2:
            st.metric("Portfolio ROI", f"{roi:.2f}%")
        
        with col3:
            profit_margin = (total_annual_profit / (portfolio_size * 1000 * base_price)) * 100
            st.metric("Profit Margin", f"{profit_margin:.2f}%")
        
        with col4:
            best_mape = df.loc[df['net_profit'].idxmax(), 'mape_scenario']
            st.metric("Optimal MAPE", f"{best_mape}%")
        
        # Financial performance by scenario
        financial_summary = df.groupby('mape_scenario').agg({
            'net_profit': ['sum', 'mean', 'std'],
            'temporal_arbitrage': 'sum',
            'zonal_arbitrage': 'sum',
            'tou_arbitrage': 'sum',
            'penalty_cost': 'sum',
            'operational_cost': 'sum'
        }).round(0)
        
        financial_summary.columns = ['_'.join(col).strip() for col in financial_summary.columns]
        financial_summary = financial_summary.reset_index()
        
        # Add ROI calculation
        financial_summary['roi_pct'] = (financial_summary['net_profit_sum'] * 365/3 / portfolio_value) * 100
        
        st.subheader("📊 Financial Summary by MAPE Scenario")
        st.dataframe(financial_summary, use_container_width=True)
        
        # Profit attribution waterfall
        st.subheader("💰 Profit Attribution Analysis")
        
        best_scenario_data = df[df['mape_scenario'] == best_mape]
        avg_profits = best_scenario_data[['temporal_arbitrage', 'zonal_arbitrage', 'tou_arbitrage',
                                         'penalty_cost', 'operational_cost']].mean()
        
        # Waterfall chart data
        categories = ['Temporal', 'Zonal', 'TOU', 'Penalties', 'Operational', 'Net']
        values = [
            avg_profits['temporal_arbitrage']/1000,
            avg_profits['zonal_arbitrage']/1000, 
            avg_profits['tou_arbitrage']/1000,
            -avg_profits['penalty_cost']/1000,
            -avg_profits['operational_cost']/1000,
            best_scenario_data['net_profit'].mean()/1000
        ]
        
        fig = go.Figure(go.Waterfall(
            name="Profit Components",
            orientation="v",
            measure=["relative", "relative", "relative", "relative", "relative", "total"],
            x=categories,
            y=values,
            text=[f"¥{v:.1f}k" for v in values],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "green"}},
            decreasing={"marker": {"color": "red"}},
            totals={"marker": {"color": "blue"}}
        ))
        
        fig.update_layout(title=f"Profit Waterfall - Best Scenario (MAPE {best_mape}%)",
                         yaxis_title="Profit Contribution (¥k)", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance comparison table
        st.subheader("📈 Scenario Comparison")
        
        comparison_data = []
        for mape in mape_scenarios:
            scenario_data = df[df['mape_scenario'] == mape]
            annual_profit = scenario_data['net_profit'].sum() * 365/3
            annual_roi = (annual_profit / portfolio_value) * 100
            avg_daily = scenario_data['net_profit'].mean()
            volatility = scenario_data['net_profit'].std()
            sharpe = avg_daily / volatility if volatility > 0 else 0
            
            comparison_data.append({
                'MAPE (%)': f"{mape}%",
                'Annual Profit (¥M)': f"{annual_profit/1e6:.2f}",
                'ROI (%)': f"{annual_roi:.2f}%",
                'Daily Avg (¥k)': f"{avg_daily/1000:.1f}",
                'Volatility (¥k)': f"{volatility/1000:.1f}",
                'Sharpe Ratio': f"{sharpe:.2f}",
                'Risk Rating': 'Low' if mape <= 2 else 'Medium' if mape <= 3.5 else 'High'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)

else:
    # Initial state - show introduction
    st.info("👈 Configure your simulation parameters in the sidebar and click 'Run 3-Day Simulation' to begin!")
    
    st.markdown("""
    ## 🎯 What This Simulator Shows
    
    This comprehensive tool demonstrates:
    
    ### 📊 **3-Day Dynamic Analysis**
    - **t-1**: Historical day with baseline conditions
    - **t**: Current day with peak demand stress test  
    - **t+1**: Future day with different weather patterns
    
    ### ⚡ **Arbitrage Strategies**
    - **Temporal Arbitrage**: Contract vs spot price exploitation
    - **Zonal Arbitrage**: Geographic price differences (North vs South Jiangsu)
    - **Time-of-Use**: Peak/valley optimization with demand response
    
    ### ⚠️ **Risk Management**
    - **Value at Risk (VaR)**: Quantified downside risk
    - **Sharpe Ratio**: Risk-adjusted returns
    - **Risk Limit Monitoring**: Daily exposure controls
    
    ### 🎛️ **Key Variables**
    - **Portfolio Size**: Scale of operations (GWh/year)
    - **MAPE Scenarios**: Forecast accuracy levels to compare
    - **Market Volatility**: Price movement intensity
    - **Risk Limits**: Maximum acceptable daily losses
    
    ### 💡 **Key Insights You'll Discover**
    - Exact profit impact of forecast accuracy
    - Which strategies are most sensitive to MAPE
    - Risk-return tradeoffs across scenarios
    - Optimal forecast accuracy targets
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    <p>Jiangsu Energy Arbitrage Simulator | Advanced Risk Management & 3-Day Dynamic Analysis</p>
    <p>Comprehensive simulation of forecast accuracy impact on energy trading profitability</p>
</div>
""", unsafe_allow_html=True)