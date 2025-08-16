import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def examine_excel_structure(file_path):
    """Examine the structure of the Excel file"""
    print(f"Examining file: {file_path}")
    
    # Read all sheets
    excel_file = pd.ExcelFile(file_path)
    print(f"Number of sheets: {len(excel_file.sheet_names)}")
    print(f"Sheet names: {excel_file.sheet_names}")
    
    for sheet_name in excel_file.sheet_names:
        print(f"\n=== Sheet: {sheet_name} ===")
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        print(f"First few rows:")
        print(df.head(3))
        print(f"Missing values: {df.isnull().sum().sum()}")

def clean_and_structure_data(file_path):
    """Clean and structure the data for ML model development"""
    
    # Read both sheets
    excel_file = pd.ExcelFile(file_path)
    
    # Read market operations data (first sheet)
    operations_df = pd.read_excel(file_path, sheet_name=excel_file.sheet_names[0])
    
    # Read pricing data (second sheet)  
    pricing_df = pd.read_excel(file_path, sheet_name=excel_file.sheet_names[1])
    
    print("Original data shapes:")
    print(f"Operations data: {operations_df.shape}")
    print(f"Pricing data: {pricing_df.shape}")
    
    # Data cleaning steps
    print("\n=== Data Cleaning ===")
    
    # 1. Handle datetime columns and duplicates
    # Use the first column as datetime index for both sheets
    operations_df.set_index(operations_df.columns[0], inplace=True)
    pricing_df.set_index(pricing_df.columns[0], inplace=True)
    
    # Remove any remaining unnamed columns
    operations_df = operations_df.loc[:, ~operations_df.columns.str.contains('^Unnamed')]
    pricing_df = pricing_df.loc[:, ~pricing_df.columns.str.contains('^Unnamed')]
    
    print(f"Operations missing values: {operations_df.isnull().sum().sum()}")
    print(f"Pricing missing values: {pricing_df.isnull().sum().sum()}")
    
    # 2. Merge the datasets on datetime index
    combined_df = pd.merge(operations_df, pricing_df, left_index=True, right_index=True, how='inner')
    
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Combined columns: {list(combined_df.columns)}")
    
    # 4. Data validation and cleaning
    # Remove any completely empty rows or columns
    combined_df = combined_df.dropna(how='all')
    combined_df = combined_df.dropna(axis=1, how='all')
    
    # 5. Handle outliers (simple approach - can be refined)
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    
    print(f"\nNumeric columns: {len(numeric_cols)}")
    print("Basic statistics:")
    print(combined_df[numeric_cols].describe())
    
    # 6. Feature engineering for time series
    if 'datetime' in combined_df.index.names or isinstance(combined_df.index, pd.DatetimeIndex):
        combined_df['hour'] = combined_df.index.hour
        combined_df['day_of_week'] = combined_df.index.dayofweek
        combined_df['month'] = combined_df.index.month
        combined_df['quarter_hour'] = (combined_df.index.minute // 15)
        
        # Peak/off-peak indicators (typical for energy markets)
        combined_df['is_peak'] = combined_df['hour'].apply(lambda x: 1 if 9 <= x <= 21 else 0)
        combined_df['is_weekend'] = combined_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    return combined_df

def export_data(df, base_path):
    """Export cleaned data to CSV and Parquet formats"""
    
    # Ensure output directory exists
    output_dir = Path(base_path) / 'cleaned_data'
    output_dir.mkdir(exist_ok=True)
    
    # Export to CSV
    csv_path = output_dir / 'energy_data_cleaned.csv'
    df.to_csv(csv_path)
    print(f"Exported to CSV: {csv_path}")
    print(f"CSV size: {csv_path.stat().st_size / 1024:.2f} KB")
    
    # Export to Parquet
    parquet_path = output_dir / 'energy_data_cleaned.parquet'
    df.to_parquet(parquet_path)
    print(f"Exported to Parquet: {parquet_path}")
    print(f"Parquet size: {parquet_path.stat().st_size / 1024:.2f} KB")
    
    # Export data dictionary
    data_dict_path = output_dir / 'data_dictionary.txt'
    with open(data_dict_path, 'w', encoding='utf-8') as f:
        f.write("Energy Trading Data Dictionary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset shape: {df.shape}\n")
        f.write(f"Time range: {df.index.min()} to {df.index.max()}\n")
        f.write(f"Frequency: 15-minute intervals\n\n")
        
        f.write("Columns:\n")
        for i, col in enumerate(df.columns, 1):
            f.write(f"{i:2d}. {col}\n")
            
        f.write(f"\nData types:\n{df.dtypes}\n")
        f.write(f"\nBasic statistics:\n{df.describe()}\n")
    
    print(f"Data dictionary saved: {data_dict_path}")
    
    return csv_path, parquet_path

if __name__ == "__main__":
    # File path
    excel_path = "input/july2025.xlsx"
    
    # Step 1: Examine structure
    print("Step 1: Examining Excel file structure...")
    examine_excel_structure(excel_path)
    
    print("\n" + "="*50)
    print("Step 2: Cleaning and structuring data...")
    
    # Step 2: Clean and structure
    cleaned_df = clean_and_structure_data(excel_path)
    
    print("\n" + "="*50)
    print("Step 3: Exporting cleaned data...")
    
    # Step 3: Export
    csv_path, parquet_path = export_data(cleaned_df, ".")
    
    print("\n" + "="*50)
    print("Data processing completed successfully!")
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    print("Files generated:")
    print(f"  - CSV: {csv_path}")
    print(f"  - Parquet: {parquet_path}")
    print("  - Data dictionary: cleaned_data/data_dictionary.txt")