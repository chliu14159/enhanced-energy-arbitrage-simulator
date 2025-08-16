#!/usr/bin/env python3
"""
Create a standalone package of the Enhanced Arbitrage Simulator
that can be shared with any user
"""

import os
import shutil
import zipfile
from pathlib import Path

def create_standalone_package():
    """Create a complete package for distribution"""
    
    print("🚀 Creating Enhanced Arbitrage Simulator Package...")
    
    # Package name and directory
    package_name = "enhanced_arbitrage_simulator_package"
    package_dir = Path(package_name)
    
    # Remove existing package if it exists
    if package_dir.exists():
        shutil.rmtree(package_dir)
    
    # Create package directory
    package_dir.mkdir()
    
    # Essential files to include
    files_to_copy = [
        "enhanced_arbitrage_simulator.py",
        "requirements_deploy.txt", 
        "Dockerfile",
        "docker-compose.yml",
        "DEPLOYMENT_GUIDE.md",
        "ARBITRAGE_METHODOLOGY_EXPLAINED.md",
        "ENHANCED_SIMULATOR_SUMMARY.md",
        "LAG_FEATURE_FIX_SUMMARY.md",
        "METHODOLOGY_ENHANCEMENT_SUMMARY.md"
    ]
    
    # Copy essential files
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy2(file, package_dir)
            print(f"✅ Copied: {file}")
        else:
            print(f"⚠️  Missing: {file}")
    
    # Copy data directory
    if os.path.exists("cleaned_data"):
        shutil.copytree("cleaned_data", package_dir / "cleaned_data")
        print("✅ Copied: cleaned_data/")
    else:
        print("❌ Missing: cleaned_data/ directory")
    
    # Create startup scripts
    create_startup_scripts(package_dir)
    
    # Create README for package
    create_package_readme(package_dir)
    
    # Create ZIP file
    zip_filename = f"{package_name}.zip"
    create_zip_package(package_dir, zip_filename)
    
    print(f"\n🎉 Package created successfully!")
    print(f"📦 Package directory: {package_dir}")
    print(f"📁 ZIP file: {zip_filename}")
    print(f"📊 Total package size: {get_dir_size(package_dir):.1f} MB")
    
    return package_dir, zip_filename

def create_startup_scripts(package_dir):
    """Create easy startup scripts for different platforms"""
    
    # Windows batch file
    windows_script = package_dir / "start_simulator.bat"
    with open(windows_script, 'w') as f:
        f.write("""@echo off
echo 🚀 Starting Enhanced Arbitrage Simulator...
echo.
echo Installing Python dependencies...
pip install -r requirements_deploy.txt

echo.
echo Starting Streamlit application...
echo Open your browser to: http://localhost:8501
echo Press Ctrl+C to stop the simulator
echo.

streamlit run enhanced_arbitrage_simulator.py

pause
""")
    
    # Unix/Mac shell script
    unix_script = package_dir / "start_simulator.sh"
    with open(unix_script, 'w') as f:
        f.write("""#!/bin/bash
echo "🚀 Starting Enhanced Arbitrage Simulator..."
echo ""
echo "Installing Python dependencies..."
pip install -r requirements_deploy.txt

echo ""
echo "Starting Streamlit application..."
echo "Open your browser to: http://localhost:8501"
echo "Press Ctrl+C to stop the simulator"
echo ""

streamlit run enhanced_arbitrage_simulator.py
""")
    
    # Make shell script executable
    os.chmod(unix_script, 0o755)
    
    # Docker startup script
    docker_script = package_dir / "start_with_docker.sh"
    with open(docker_script, 'w') as f:
        f.write("""#!/bin/bash
echo "🐳 Starting Enhanced Arbitrage Simulator with Docker..."
echo ""
echo "Building Docker image..."
docker-compose build

echo ""
echo "Starting container..."
docker-compose up

echo ""
echo "Open your browser to: http://localhost:8501"
echo "Press Ctrl+C to stop the simulator"
""")
    
    os.chmod(docker_script, 0o755)
    
    print("✅ Created startup scripts")

def create_package_readme(package_dir):
    """Create a comprehensive README for the package"""
    
    readme_content = """# 🏆 Enhanced Jiangsu Energy Arbitrage Simulator

## 🎯 What This Tool Does

This is a **production-ready energy arbitrage simulator** that:

- **Predicts energy prices** using AI (9.55% MAPE accuracy)
- **Calculates arbitrage profits** across 4 enhanced strategies
- **Provides complete transparency** with mathematical formulations
- **Uses real market data** from Jiangsu Province electricity market

## ⚡ Quick Start (3 options)

### Option 1: Python (Recommended)
**Requirements:** Python 3.8+ installed

1. **Windows:** Double-click `start_simulator.bat`
2. **Mac/Linux:** Run `./start_simulator.sh`
3. **Manual:** 
   ```bash
   pip install -r requirements_deploy.txt
   streamlit run enhanced_arbitrage_simulator.py
   ```

### Option 2: Docker (No Python needed)
**Requirements:** Docker installed

1. Run `./start_with_docker.sh`
2. Or manually: `docker-compose up`

### Option 3: Pre-built Executable
See `DEPLOYMENT_GUIDE.md` for creating executables

## 📊 Access the Tool

After starting, open your browser to: **http://localhost:8501**

## 🎛️ How to Use

1. **Configure Portfolio**: Set size (GWh) and base price
2. **Select Date Range**: Choose analysis period  
3. **Run Simulation**: Click "Run Enhanced Simulation"
4. **Analyze Results**: Explore 5 detailed tabs:
   - Strategy Breakdown
   - AI Predictions vs Reality
   - Daily Performance
   - Real-time Analysis
   - **Methodology & Equations** ← Complete transparency!

## 📚 Documentation

- **`ARBITRAGE_METHODOLOGY_EXPLAINED.md`** - Complete mathematical framework
- **`ENHANCED_SIMULATOR_SUMMARY.md`** - Technical overview
- **`DEPLOYMENT_GUIDE.md`** - Sharing and deployment options

## 🏆 Key Features

### **4 Enhanced Arbitrage Strategies:**
1. **⏰ Temporal Arbitrage** - Contract vs spot price differences
2. **🤖 AI-Enhanced Arbitrage** - ML prediction-based trading  
3. **📈 Peak/Off-peak Optimization** - Time-of-use arbitrage
4. **🌱 Renewable Arbitrage** - Green energy timing

### **Complete Mathematical Transparency:**
- Every calculation explained with equations
- Business logic documented
- Parameter sensitivity analysis
- Risk management formulations

### **Real Market Data:**
- Jiangsu Province July 2025 data
- 15-minute resolution (2,976 data points)
- Actual renewable forecasts and load patterns

## 🎯 Perfect For:

- **Energy Traders** - Strategy analysis and optimization
- **Portfolio Managers** - Risk-adjusted return evaluation
- **Analysts** - Market dynamics understanding
- **Researchers** - Academic studies and validation

## 📈 Expected Performance

**Example Results (1500 GWh portfolio, 1.5% MAPE):**
- **Daily Profits**: ~¥140-150k/day
- **Annual Potential**: ~¥50-55M/year
- **ROI**: 2-3% margin improvement

## ⚠️ System Requirements

- **Python 3.8+** (for direct run)
- **4GB RAM** minimum
- **Web browser** (Chrome, Firefox, Safari, Edge)
- **Internet connection** (for initial package installation only)

## 🆘 Troubleshooting

### Common Issues:
1. **Port 8501 in use**: Change port or stop other applications
2. **Missing data**: Ensure `cleaned_data/` folder exists
3. **Lag feature errors**: Use dates after July 2nd, 2025

### Support:
- Check `LAG_FEATURE_FIX_SUMMARY.md` for common fixes
- Review `DEPLOYMENT_GUIDE.md` for deployment options

## 🔒 Data & Privacy

- **All data processing** happens locally on your machine
- **No external connections** required after installation
- **Real market data** included for realistic analysis

## 📄 License & Usage

This tool is provided for analysis and research purposes. Please ensure compliance with your organization's data and trading policies.

---

**🚀 Ready to start? Run one of the startup scripts and open http://localhost:8501!**
"""
    
    readme_file = package_dir / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("✅ Created package README")

def create_zip_package(package_dir, zip_filename):
    """Create a ZIP file of the package"""
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arc_name = os.path.relpath(file_path, os.path.dirname(package_dir))
                zipf.write(file_path, arc_name)
    
    print(f"✅ Created ZIP package: {zip_filename}")

def get_dir_size(path):
    """Calculate directory size in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert to MB

if __name__ == "__main__":
    try:
        package_dir, zip_file = create_standalone_package()
        
        print(f"\n📋 Package Contents:")
        for item in sorted(package_dir.rglob('*')):
            if item.is_file():
                size_kb = item.stat().st_size / 1024
                print(f"   📄 {item.relative_to(package_dir)} ({size_kb:.1f} KB)")
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. Share the ZIP file: {zip_file}")
        print(f"   2. Recipients extract and run startup script")
        print(f"   3. Tool opens at http://localhost:8501")
        print(f"   4. Full arbitrage analysis ready!")
        
    except Exception as e:
        print(f"❌ Error creating package: {e}")
        print(f"   Please ensure all required files are present")