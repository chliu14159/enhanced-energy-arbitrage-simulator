#!/usr/bin/env python3
"""
Wind and Solar Data Collection and Cleaning Script
=================================================

This script processes wind and solar energy data from SQL files and Excel metadata
to create a clean, consolidated dataset for modeling.

Author: GitHub Copilot Assistant
Date: 22 August 2025
"""

import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WindSolarDataProcessor:
    """Process wind and solar data from SQL dumps and Excel metadata."""
    
    def __init__(self, input_dir: str, output_dir: str = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir.parent / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.metadata = {}
        self.raw_data = None
        self.cleaned_data = None
        
        logger.info(f"Initialized processor with input: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def read_excel_metadata(self, excel_file: str = "光伏统计.xlsx"):
        """Read field definitions from Excel file."""
        excel_path = self.input_dir / excel_file
        
        if not excel_path.exists():
            logger.warning(f"Excel metadata file not found: {excel_path}")
            return
        
        try:
            # Try to read all sheets
            excel_data = pd.read_excel(excel_path, sheet_name=None, engine='openpyxl')
            
            logger.info(f"Found {len(excel_data)} sheets in Excel file:")
            for sheet_name in excel_data.keys():
                logger.info(f"  - {sheet_name}")
                df = excel_data[sheet_name]
                logger.info(f"    Shape: {df.shape}")
                
                # Display first few rows to understand structure
                logger.info(f"    First 3 rows:\n{df.head(3)}")
                
            self.metadata = excel_data
            return excel_data
            
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            return None
    
    def parse_sql_file(self, sql_file_path: Path):
        """Parse a single SQL file and extract data."""
        logger.info(f"Processing SQL file: {sql_file_path.name}")
        
        # Extract station ID from filename
        station_id = sql_file_path.stem
        
        data_rows = []
        
        try:
            with open(sql_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line_num % 10000 == 0 and line_num > 0:
                        logger.info(f"  Processed {line_num:,} lines...")
                    
                    # Parse INSERT statements using regex
                    match = re.search(
                        r"insert into.*?\((.*?)\)\s+values\s+\((.*?)\);", 
                        line.strip(), 
                        re.IGNORECASE
                    )
                    
                    if match:
                        columns = [col.strip().strip('"') for col in match.group(1).split(',')]
                        values = match.group(2).split(',')
                        
                        # Clean up values
                        cleaned_values = []
                        for val in values:
                            val = val.strip().strip("'\"")
                            # Convert to appropriate type
                            if val.replace('.', '').replace('-', '').isdigit():
                                cleaned_values.append(float(val) if '.' in val else int(val))
                            else:
                                cleaned_values.append(val)
                        
                        # Create row dictionary
                        if len(columns) == len(cleaned_values):
                            row = dict(zip(columns, cleaned_values))
                            row['station_id'] = station_id
                            data_rows.append(row)
        
        except Exception as e:
            logger.error(f"Error parsing {sql_file_path.name}: {e}")
            return None
        
        if data_rows:
            df = pd.DataFrame(data_rows)
            logger.info(f"  Extracted {len(df):,} records with columns: {list(df.columns)}")
            return df
        else:
            logger.warning(f"  No data extracted from {sql_file_path.name}")
            return None
    
    def process_all_sql_files(self):
        """Process all SQL files in the input directory."""
        sql_files = list(self.input_dir.glob("*.sql"))
        logger.info(f"Found {len(sql_files)} SQL files to process")
        
        all_data = []
        
        for sql_file in sql_files:
            df = self.parse_sql_file(sql_file)
            if df is not None:
                all_data.append(df)
        
        if all_data:
            # Combine all dataframes
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Combined data shape: {combined_df.shape}")
            
            self.raw_data = combined_df
            return combined_df
        else:
            logger.error("No data was successfully extracted from SQL files")
            return None
    
    def clean_and_process_data(self):
        """Clean and process the raw data."""
        if self.raw_data is None or len(self.raw_data) == 0:
            logger.error("No raw data available for cleaning")
            return None
        
        df = self.raw_data.copy()
        logger.info(f"Starting data cleaning for {len(df):,} records")
        
        # 1. Parse timestamp
        if 'OCCUR_TIME' in df.columns:
            df['datetime'] = pd.to_datetime(df['OCCUR_TIME'])
            df['date'] = df['datetime'].dt.date
            df['hour'] = df['datetime'].dt.hour
            df['minute'] = df['datetime'].dt.minute
            logger.info(f"  Parsed timestamps: {df['datetime'].min()} to {df['datetime'].max()}")
        
        # 2. Identify and clean measurement columns
        measurement_cols = [col for col in df.columns if col.startswith('CUR_')]
        status_cols = [col for col in df.columns if col.startswith('STA_')]
        
        logger.info(f"  Found {len(measurement_cols)} measurement columns: {measurement_cols}")
        logger.info(f"  Found {len(status_cols)} status columns: {status_cols}")
        
        # 3. Analyze data by station
        station_analysis = []
        for station_id in df['station_id'].unique():
            station_data = df[df['station_id'] == station_id]
            
            analysis = {
                'station_id': station_id,
                'record_count': len(station_data),
                'date_range': f"{station_data['datetime'].min()} to {station_data['datetime'].max()}",
                'measurement_cols': [col for col in measurement_cols if col in station_data.columns],
                'status_cols': [col for col in status_cols if col in station_data.columns]
            }
            
            # Analyze measurement patterns
            for col in analysis['measurement_cols']:
                values = station_data[col]
                # Convert to numeric, replacing non-numeric values with NaN
                numeric_values = pd.to_numeric(values, errors='coerce')
                analysis[f'{col}_stats'] = {
                    'min': numeric_values.min(),
                    'max': numeric_values.max(),
                    'mean': numeric_values.mean(),
                    'zero_ratio': (numeric_values == 0).mean(),
                    'non_zero_count': (numeric_values != 0).sum(),
                    'null_count': numeric_values.isna().sum()
                }
            
            station_analysis.append(analysis)
            logger.info(f"  Station {station_id}: {len(station_data):,} records, "
                       f"cols: {analysis['measurement_cols']}")
        
        # 4. Create cleaned dataset with meaningful column names
        # Based on the data patterns observed:
        # - 501974: CUR_090 (small values 0.3-0.4, likely wind speed or capacity factor)
        # - 502633: CUR_040 (all zeros, likely solar at night)
        # - 505519: CUR_092 (all zeros, likely solar at night) 
        # - 506445: CUR_024 (large values 7000+, likely actual power output)
        
        station_mapping = {
            '501974': {'type': 'wind', 'measurement': 'wind_speed_or_factor', 'unit': 'unknown'},
            '502633': {'type': 'solar', 'measurement': 'solar_power', 'unit': 'MW'},
            '505519': {'type': 'solar', 'measurement': 'solar_power', 'unit': 'MW'},
            '506445': {'type': 'wind', 'measurement': 'wind_power', 'unit': 'MW'}
        }
        
        # 5. Restructure data for modeling
        cleaned_records = []
        
        for _, row in df.iterrows():
            station_id = row['station_id']
            
            # Get measurement columns for this station
            station_measurement_cols = [col for col in measurement_cols if col in row.index and pd.notna(row[col])]
            
            for col in station_measurement_cols:
                # Convert value to numeric
                value = pd.to_numeric(row[col], errors='coerce')
                if pd.isna(value):
                    continue  # Skip non-numeric values
                
                record = {
                    'datetime': row['datetime'],
                    'date': row['date'],
                    'hour': row['hour'],
                    'minute': row['minute'],
                    'station_id': station_id,
                    'measurement_type': col,
                    'value': value,
                    'status': row.get(col.replace('CUR_', 'STA_'), None)
                }
                
                # Add metadata if available
                if station_id in station_mapping:
                    record.update(station_mapping[station_id])
                
                cleaned_records.append(record)
        
        cleaned_df = pd.DataFrame(cleaned_records)
        
        # 6. Additional cleaning
        # Remove invalid timestamps
        cleaned_df = cleaned_df.dropna(subset=['datetime'])
        
        # Remove obviously invalid values (negative power)
        cleaned_df = cleaned_df[cleaned_df['value'] >= 0]
        
        # Sort by datetime and station
        cleaned_df = cleaned_df.sort_values(['datetime', 'station_id'])
        
        logger.info(f"Cleaned data shape: {cleaned_df.shape}")
        logger.info(f"Date range: {cleaned_df['datetime'].min()} to {cleaned_df['datetime'].max()}")
        logger.info(f"Unique stations: {cleaned_df['station_id'].nunique()}")
        
        self.cleaned_data = cleaned_df
        return cleaned_df
    
    def save_data(self, save_formats=['csv', 'parquet']):
        """Save the cleaned data in specified formats."""
        if self.cleaned_data is None:
            logger.error("No cleaned data available to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"wind_solar_data_cleaned_{timestamp}"
        
        saved_files = []
        
        if 'csv' in save_formats:
            csv_path = self.output_dir / f"{base_filename}.csv"
            self.cleaned_data.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV: {csv_path}")
            saved_files.append(csv_path)
        
        if 'parquet' in save_formats:
            parquet_path = self.output_dir / f"{base_filename}.parquet"
            self.cleaned_data.to_parquet(parquet_path, index=False)
            logger.info(f"Saved Parquet: {parquet_path}")
            saved_files.append(parquet_path)
        
        # Save metadata and summary
        summary_path = self.output_dir / f"data_summary_{timestamp}.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Wind and Solar Data Processing Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Processing Date: {datetime.now()}\n")
            f.write(f"Total Records: {len(self.cleaned_data):,}\n")
            f.write(f"Date Range: {self.cleaned_data['datetime'].min()} to {self.cleaned_data['datetime'].max()}\n")
            f.write(f"Unique Stations: {self.cleaned_data['station_id'].nunique()}\n\n")
            
            f.write("Station Summary:\n")
            for station in self.cleaned_data['station_id'].unique():
                station_data = self.cleaned_data[self.cleaned_data['station_id'] == station]
                f.write(f"  {station}: {len(station_data):,} records\n")
            
            f.write("\nColumn Information:\n")
            for col in self.cleaned_data.columns:
                f.write(f"  {col}: {self.cleaned_data[col].dtype}\n")
        
        logger.info(f"Saved summary: {summary_path}")
        saved_files.append(summary_path)
        
        return saved_files
    
    def generate_eda_report(self):
        """Generate basic EDA report."""
        if self.cleaned_data is None:
            logger.error("No cleaned data available for EDA")
            return
        
        df = self.cleaned_data
        
        print("\n" + "="*60)
        print("WIND AND SOLAR DATA - EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   Total Records: {len(df):,}")
        print(f"   Date Range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"   Unique Stations: {df['station_id'].nunique()}")
        print(f"   Data Types: {df['type'].value_counts().to_dict()}")
        
        print(f"\n🏭 STATION ANALYSIS:")
        for station in df['station_id'].unique():
            station_data = df[df['station_id'] == station]
            station_type = station_data['type'].iloc[0] if 'type' in station_data.columns else 'unknown'
            
            print(f"   Station {station} ({station_type}):")
            print(f"     Records: {len(station_data):,}")
            print(f"     Value Range: {station_data['value'].min():.2f} - {station_data['value'].max():.2f}")
            print(f"     Non-zero Records: {(station_data['value'] > 0).sum():,} ({(station_data['value'] > 0).mean()*100:.1f}%)")
        
        print(f"\n📈 VALUE STATISTICS:")
        print(df.groupby(['station_id', 'type'])['value'].agg(['count', 'mean', 'std', 'min', 'max']).round(2))
        
        print(f"\n⏰ TEMPORAL PATTERNS:")
        hourly_avg = df.groupby(['station_id', 'hour'])['value'].mean().unstack(level=0)
        print("Average values by hour:")
        print(hourly_avg.round(2))


def main():
    """Main execution function."""
    print("🌪️ Wind and Solar Data Processing Script")
    print("=" * 50)
    
    # Set up paths
    input_dir = "/Users/randomwalk/Documents/CODE/REPO/energy_trading_js/input/wind_and_solar"
    output_dir = "/Users/randomwalk/Documents/CODE/REPO/energy_trading_js/processed/wind_solar"
    
    # Initialize processor
    processor = WindSolarDataProcessor(input_dir, output_dir)
    
    # Step 1: Read Excel metadata
    print("\n📋 Step 1: Reading Excel metadata...")
    metadata = processor.read_excel_metadata()
    
    # Step 2: Process SQL files
    print("\n🔄 Step 2: Processing SQL files...")
    raw_data = processor.process_all_sql_files()
    
    if raw_data is None:
        print("❌ Failed to process SQL files. Exiting.")
        return
    
    # Step 3: Clean and process data
    print("\n🧹 Step 3: Cleaning and processing data...")
    cleaned_data = processor.clean_and_process_data()
    
    if cleaned_data is None:
        print("❌ Failed to clean data. Exiting.")
        return
    
    # Step 4: Save processed data
    print("\n💾 Step 4: Saving processed data...")
    saved_files = processor.save_data(['csv', 'parquet'])
    
    # Step 5: Generate EDA report
    print("\n📊 Step 5: Generating EDA report...")
    processor.generate_eda_report()
    
    print("\n✅ Processing complete!")
    if saved_files:
        print(f"Saved files:")
        for file_path in saved_files:
            print(f"  - {file_path}")
    else:
        print("No files were saved.")


if __name__ == "__main__":
    main()
