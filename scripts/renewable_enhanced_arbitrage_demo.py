#!/usr/bin/env python3
"""
Renewable-Enhanced Arbitrage Strategy
====================================

This script demonstrates how to integrate renewable energy forecasting 
into the existing arbitrage model for improved trading strategies.

Author: GitHub Copilot Assistant
Date: 22 August 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class RenewableEnhancedArbitrage:
    """Enhanced arbitrage strategy incorporating renewable forecasting."""
    
    def __init__(self):
        # Battery configuration (from existing model)
        self.battery_capacity = 50.0  # MWh
        self.max_charge_rate = 25.0   # MW
        self.max_discharge_rate = 25.0  # MW
        self.round_trip_efficiency = 0.85
        
        # Renewable portfolio (based on our trained models)
        self.renewable_stations = {
            '501974': {'type': 'wind', 'capacity': 12, 'avg_cf': 0.201},
            '502633': {'type': 'solar', 'capacity': 32, 'avg_cf': 0.250},
            '506445': {'type': 'wind', 'capacity': 20, 'avg_cf': 0.295}
        }
        
        print("🔋 Renewable-Enhanced Arbitrage Model Initialized")
        print(f"📊 Battery: {self.battery_capacity} MWh capacity")
        print(f"🌱 Renewable Portfolio: {sum(s['capacity'] for s in self.renewable_stations.values())} MW total")
    
    def simulate_renewable_generation(self, hours=24):
        """Simulate renewable generation patterns for demonstration."""
        time_range = pd.date_range(
            start=datetime.now(), 
            periods=hours*4,  # 15-minute intervals
            freq='15min'
        )
        
        generation_data = []
        
        for timestamp in time_range:
            hour = timestamp.hour
            
            for station_id, config in self.renewable_stations.items():
                if config['type'] == 'solar':
                    # Solar generation pattern (0 at night, peak at noon)
                    if 6 <= hour <= 18:
                        solar_factor = np.sin(np.pi * (hour - 6) / 12)
                        generation = config['capacity'] * solar_factor * config['avg_cf'] * 4
                    else:
                        generation = 0
                else:  # wind
                    # Wind generation with some variability
                    base_generation = config['capacity'] * config['avg_cf']
                    variability = np.random.normal(0, base_generation * 0.3)
                    generation = max(0, base_generation + variability)
                
                generation_data.append({
                    'datetime': timestamp,
                    'station_id': station_id,
                    'type': config['type'],
                    'generation': generation
                })
        
        return pd.DataFrame(generation_data)
    
    def simulate_load_forecast(self, hours=24):
        """Simulate load forecast for demonstration."""
        time_range = pd.date_range(
            start=datetime.now(), 
            periods=hours*4,  # 15-minute intervals
            freq='15min'
        )
        
        load_data = []
        base_load = 1000  # MW
        
        for timestamp in time_range:
            hour = timestamp.hour
            
            # Typical daily load pattern
            if 0 <= hour < 6:
                load_factor = 0.7  # Night
            elif 6 <= hour < 9:
                load_factor = 0.9  # Morning ramp
            elif 9 <= hour < 17:
                load_factor = 1.0  # Day
            elif 17 <= hour < 22:
                load_factor = 1.1  # Evening peak
            else:
                load_factor = 0.8  # Late evening
            
            # Add some randomness
            load = base_load * load_factor * (1 + np.random.normal(0, 0.1))
            
            load_data.append({
                'datetime': timestamp,
                'load': load
            })
        
        return pd.DataFrame(load_data)
    
    def calculate_net_load(self, load_df, renewable_df):
        """Calculate net load after accounting for renewable generation."""
        # Aggregate renewable generation by timestamp
        renewable_agg = renewable_df.groupby('datetime')['generation'].sum().reset_index()
        renewable_agg.columns = ['datetime', 'total_renewable']
        
        # Merge with load data
        combined = pd.merge(load_df, renewable_agg, on='datetime', how='left')
        combined['total_renewable'] = combined['total_renewable'].fillna(0)
        
        # Calculate net load
        combined['net_load'] = combined['load'] - combined['total_renewable']
        
        return combined
    
    def enhanced_arbitrage_strategy(self, combined_df, price_premium=50):
        """
        Enhanced arbitrage strategy considering renewables.
        
        Args:
            combined_df: DataFrame with load, renewable, and net_load
            price_premium: Price difference for arbitrage ($/MWh)
        """
        results = []
        battery_soc = 0.5  # Start at 50% state of charge
        
        for idx, row in combined_df.iterrows():
            timestamp = row['datetime']
            net_load = row['net_load']
            renewable_gen = row['total_renewable']
            load = row['load']
            
            # Simple price model based on net load
            base_price = 100  # $/MWh
            price = base_price + (net_load - 800) * 0.1  # Price increases with net load
            
            # Arbitrage decision logic
            action = 'hold'
            power = 0
            
            # High renewable generation + low net load = charging opportunity
            if renewable_gen > 30 and net_load < 900 and battery_soc < 0.9:
                action = 'charge'
                power = min(self.max_charge_rate, 
                           (0.9 - battery_soc) * self.battery_capacity)
            
            # Low renewable generation + high net load = discharging opportunity
            elif renewable_gen < 10 and net_load > 1100 and battery_soc > 0.1:
                action = 'discharge'
                power = min(self.max_discharge_rate,
                           (battery_soc - 0.1) * self.battery_capacity)
            
            # Traditional arbitrage (low price = charge, high price = discharge)
            elif price < 80 and battery_soc < 0.9:
                action = 'charge'
                power = min(self.max_charge_rate * 0.5,  # Conservative charging
                           (0.9 - battery_soc) * self.battery_capacity)
            
            elif price > 120 and battery_soc > 0.1:
                action = 'discharge'
                power = min(self.max_discharge_rate * 0.5,  # Conservative discharging
                           (battery_soc - 0.1) * self.battery_capacity)
            
            # Update battery state of charge
            if action == 'charge':
                battery_soc += (power * 0.25 * self.round_trip_efficiency) / self.battery_capacity
            elif action == 'discharge':
                battery_soc -= (power * 0.25) / self.battery_capacity
            
            # Ensure SOC stays within bounds
            battery_soc = max(0, min(1, battery_soc))
            
            # Calculate revenue (simplified)
            if action == 'charge':
                revenue = -power * price * 0.25  # Cost to charge
            elif action == 'discharge':
                revenue = power * price * 0.25   # Revenue from discharge
            else:
                revenue = 0
            
            results.append({
                'datetime': timestamp,
                'load': load,
                'renewable_gen': renewable_gen,
                'net_load': net_load,
                'price': price,
                'action': action,
                'power': power,
                'battery_soc': battery_soc,
                'revenue': revenue
            })
        
        return pd.DataFrame(results)
    
    def analyze_results(self, strategy_df):
        """Analyze arbitrage strategy results."""
        print(f"\n📊 RENEWABLE-ENHANCED ARBITRAGE ANALYSIS")
        print("=" * 50)
        
        # Total revenue
        total_revenue = strategy_df['revenue'].sum()
        print(f"💰 Total Revenue: ${total_revenue:,.2f}")
        
        # Cycling analysis
        charge_cycles = (strategy_df['action'] == 'charge').sum()
        discharge_cycles = (strategy_df['action'] == 'discharge').sum()
        print(f"🔋 Charge Cycles: {charge_cycles}")
        print(f"⚡ Discharge Cycles: {discharge_cycles}")
        
        # Renewable utilization
        avg_renewable = strategy_df['renewable_gen'].mean()
        max_renewable = strategy_df['renewable_gen'].max()
        print(f"🌱 Average Renewable: {avg_renewable:.1f} MW")
        print(f"🌱 Peak Renewable: {max_renewable:.1f} MW")
        
        # Net load statistics
        avg_net_load = strategy_df['net_load'].mean()
        print(f"📈 Average Net Load: {avg_net_load:.1f} MW")
        
        # Price statistics
        avg_price = strategy_df['price'].mean()
        min_price = strategy_df['price'].min()
        max_price = strategy_df['price'].max()
        print(f"💵 Price Range: ${min_price:.2f} - ${max_price:.2f} (avg: ${avg_price:.2f})")
        
        return {
            'total_revenue': total_revenue,
            'charge_cycles': charge_cycles,
            'discharge_cycles': discharge_cycles,
            'avg_renewable': avg_renewable,
            'avg_net_load': avg_net_load,
            'avg_price': avg_price
        }
    
    def plot_results(self, strategy_df):
        """Plot strategy results."""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Load vs Net Load
        axes[0].plot(strategy_df['datetime'], strategy_df['load'], 
                    label='Total Load', linewidth=2)
        axes[0].plot(strategy_df['datetime'], strategy_df['renewable_gen'], 
                    label='Renewable Generation', linewidth=2)
        axes[0].plot(strategy_df['datetime'], strategy_df['net_load'], 
                    label='Net Load', linewidth=2)
        axes[0].set_ylabel('Power (MW)')
        axes[0].set_title('Load Profile with Renewable Generation')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Battery SOC and Actions
        axes[1].plot(strategy_df['datetime'], strategy_df['battery_soc'] * 100, 
                    label='Battery SOC (%)', linewidth=2, color='green')
        
        # Color-code actions
        charge_mask = strategy_df['action'] == 'charge'
        discharge_mask = strategy_df['action'] == 'discharge'
        
        axes[1].scatter(strategy_df.loc[charge_mask, 'datetime'], 
                       strategy_df.loc[charge_mask, 'battery_soc'] * 100,
                       color='blue', label='Charging', alpha=0.7, s=30)
        axes[1].scatter(strategy_df.loc[discharge_mask, 'datetime'], 
                       strategy_df.loc[discharge_mask, 'battery_soc'] * 100,
                       color='red', label='Discharging', alpha=0.7, s=30)
        
        axes[1].set_ylabel('Battery SOC (%)')
        axes[1].set_title('Battery State of Charge and Actions')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Price and Revenue
        axes[2].plot(strategy_df['datetime'], strategy_df['price'], 
                    label='Electricity Price', linewidth=2, color='orange')
        
        ax2_twin = axes[2].twinx()
        ax2_twin.plot(strategy_df['datetime'], strategy_df['revenue'].cumsum(), 
                     label='Cumulative Revenue', linewidth=2, color='purple')
        ax2_twin.set_ylabel('Cumulative Revenue ($)', color='purple')
        
        axes[2].set_ylabel('Price ($/MWh)', color='orange')
        axes[2].set_xlabel('Time')
        axes[2].set_title('Electricity Price and Cumulative Revenue')
        axes[2].grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('/Users/randomwalk/Documents/CODE/REPO/energy_trading_js/models/lstm_forecasting/renewable_arbitrage_demo.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig


def main():
    """Demonstrate renewable-enhanced arbitrage strategy."""
    print("🔋 RENEWABLE-ENHANCED ARBITRAGE DEMONSTRATION")
    print("=" * 60)
    
    # Initialize model
    arbitrage = RenewableEnhancedArbitrage()
    
    # Generate simulated data
    print("\n📊 Generating simulation data...")
    renewable_df = arbitrage.simulate_renewable_generation(hours=24)
    load_df = arbitrage.simulate_load_forecast(hours=24)
    
    # Calculate net load
    combined_df = arbitrage.calculate_net_load(load_df, renewable_df)
    
    # Run enhanced arbitrage strategy
    print("🚀 Running enhanced arbitrage strategy...")
    strategy_df = arbitrage.enhanced_arbitrage_strategy(combined_df)
    
    # Analyze results
    results = arbitrage.analyze_results(strategy_df)
    
    # Plot results
    print("\n📈 Generating visualizations...")
    arbitrage.plot_results(strategy_df)
    
    print(f"\n✅ Demonstration complete!")
    print(f"💡 Key Insight: Renewable generation creates additional arbitrage opportunities")
    print(f"🎯 Next Step: Integrate LSTM forecasting models for real-time predictions")


if __name__ == "__main__":
    main()
