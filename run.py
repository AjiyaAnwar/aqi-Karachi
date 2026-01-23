"""
🚀 AQI Karachi - Main Runner Script
UPDATED FOR YOUR PROJECT STRUCTURE
"""
import os
import sys
import subprocess
from datetime import datetime

def print_header():
    """Print application header"""
    print("=" * 60)
    print("🌫️ AQI KARACHI - AIR QUALITY PREDICTION SYSTEM")
    print("=" * 60)
    print("📊 Real-time AQI monitoring and 3-day forecasting")
    print("🤖 Machine Learning + Time Series models")
    print("💾 MongoDB Atlas + Open-Meteo API")
    print("=" * 60)

def run_collect_data():
    """Collect 45 days of historical data"""
    print(f"\n[{datetime.now()}] 📥 Collecting 45 days of AQI data...")
    try:
        # From your structure: data_pipeline/collect_historical.py
        subprocess.run([sys.executable, "data_pipeline/collect_historical.py"])
    except FileNotFoundError:
        print("❌ Error: collect_historical.py not found in data_pipeline/")
        print("   Available files in data_pipeline/:")
        try:
            files = os.listdir("data_pipeline")
            for f in files:
                print(f"    - {f}")
        except:
            print("    (data_pipeline directory not found)")

def run_feature_engineering():
    """Create features from raw data"""
    print(f"\n[{datetime.now()}] 🔧 Creating features...")
    try:
        # From your structure: data_pipeline/features.py
        subprocess.run([sys.executable, "data_pipeline/features.py"])
    except FileNotFoundError:
        print("❌ Error: features.py not found in data_pipeline/")
        print("   Available files in data_pipeline/:")
        try:
            files = os.listdir("data_pipeline")
            for f in files:
                print(f"    - {f}")
        except:
            print("    (data_pipeline directory not found)")

def run_ml_training():
    """Train ML models"""
    print(f"\n[{datetime.now()}] 🤖 Training ML models...")
    try:
        # From your structure: model_training/train_models.py
        subprocess.run([sys.executable, "model_training/train_models.py"])
    except FileNotFoundError:
        print("❌ Error: train_models.py not found in model_training/")
        print("   Available files in model_training/:")
        try:
            files = os.listdir("model_training")
            for f in files:
                print(f"    - {f}")
        except:
            print("    (model_training directory not found)")

def run_time_series():
    """Run time series forecasting"""
    print(f"\n[{datetime.now()}] 📈 Running time series forecasting...")
    try:
        # From your structure: model_training/train_time_series_models.py
        subprocess.run([sys.executable, "model_training/train_time_series_models.py"])
    except FileNotFoundError:
        print("❌ Error: train_time_series_models.py not found in model_training/")
        print("   Available files in model_training/:")
        try:
            files = os.listdir("model_training")
            for f in files:
                print(f"    - {f}")
        except:
            print("    (model_training directory not found)")

def run_all_models():
    """Run both ML and Time Series models"""
    print(f"\n[{datetime.now()}] 🚀 Running ALL forecasting models...")
    try:
        # From your structure: model_training/runallmodels.py
        subprocess.run([sys.executable, "model_training/runallmodels.py"])
    except FileNotFoundError:
        print("⚠️ runallmodels.py not found in model_training/")
        print("   Running individual models instead...")
        run_ml_training()
        run_time_series()

def run_dashboard():
    """Launch Streamlit dashboard"""
    print(f"\n[{datetime.now()}] 📊 Launching dashboard...")
    print("   🌐 Open: http://localhost:8501")
    print("   🛑 Press Ctrl+C to stop the dashboard")
    
    # Check if dashboard is in dashboard/ or root
    dashboard_paths = ["dashboard/dashboard.py", "dashboard.py"]
    
    for path in dashboard_paths:
        if os.path.exists(path):
            try:
                subprocess.run(["streamlit", "run", path])
                return
            except Exception as e:
                print(f"❌ Error running {path}: {e}")
                break
    
    print("❌ Error: dashboard.py not found!")
    print("   Checked:")
    for path in dashboard_paths:
        print(f"    - {path}")

def run_cicd():
    """Run CI/CD pipeline"""
    print(f"\n[{datetime.now()}] 🔄 Starting CI/CD pipeline...")
    
    # Check for cicd directory
    cicd_paths = ["cicd/pipeline.py", "pipeline.py"]
    
    for path in cicd_paths:
        if os.path.exists(path):
            try:
                subprocess.run([sys.executable, path])
                return
            except Exception as e:
                print(f"❌ Error running {path}: {e}")
                break
    
    print("❌ Error: CI/CD pipeline not found!")
    print("   Checked:")
    for path in cicd_paths:
        print(f"    - {path}")

def run_eda():
    """Run EDA analysis"""
    print(f"\n[{datetime.now()}] 📈 Running EDA analysis...")
    
    # Check for EDA in different locations
    eda_paths = [
        "notebook/eda.py",           # Your structure
        "notebook/eda_intelligent.py", # New intelligent EDA
        "eda.py",                    # Root
        "notebooks/eda.py"           # Common location
    ]
    
    for path in eda_paths:
        if os.path.exists(path):
            try:
                subprocess.run([sys.executable, path])
                return
            except Exception as e:
                print(f"❌ Error running {path}: {e}")
                continue
    
    print("❌ Error: EDA script not found!")
    print("   Checked paths:")
    for path in eda_paths:
        print(f"    - {path}")

def run_tests():
    """Run tests"""
    print(f"\n[{datetime.now()}] 🧪 Running tests...")
    
    test_paths = [
        "testdata.py",          # From your structure
        "tests/testdata.py",
        "test/testdata.py"
    ]
    
    for path in test_paths:
        if os.path.exists(path):
            try:
                result = subprocess.run([sys.executable, path], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ Tests passed!")
                else:
                    print("❌ Tests failed!")
                    if result.stderr:
                        print(f"Error: {result.stderr[:200]}")
                return
            except Exception as e:
                print(f"❌ Error running tests: {e}")
                break
    
    print("⚠️ No test files found. Skipping tests.")

def check_system():
    """Check system status and dependencies"""
    print(f"\n[{datetime.now()}] 🔍 Checking system status...")
    
    # Check Python version
    print(f"   Python: {sys.version.split()[0]}")
    
    # Check your exact project structure
    print("\n   📁 PROJECT STRUCTURE:")
    
    structure = {
        "data_pipeline/": ["collect_historical.py", "features.py"],
        "model_training/": ["train_models.py", "train_time_series_models.py", "runallmodels.py"],
        "dashboard/": ["dashboard.py"],
        "cicd/": ["pipeline.py"],
        "notebook/": ["eda.py"],
        "": [".env", "testdata.py", "requirements.txt", "run.py"]
    }
    
    all_good = True
    
    for directory, expected_files in structure.items():
        if directory.strip("/"):
            # Check directory
            if os.path.exists(directory):
                print(f"   ✓ {directory}")
                # Check files in directory
                try:
                    actual_files = os.listdir(directory)
                    for expected_file in expected_files:
                        if expected_file in actual_files:
                            print(f"      ✓ {expected_file}")
                        else:
                            print(f"      ⚠️ {expected_file} (missing)")
                            all_good = False
                except:
                    print(f"      (cannot read {directory})")
            else:
                print(f"   ❌ {directory} (missing)")
                all_good = False
        else:
            # Check root files
            for expected_file in expected_files:
                if os.path.exists(expected_file):
                    print(f"   ✓ {expected_file}")
                else:
                    print(f"   ⚠️ {expected_file} (missing)")
                    all_good = False
    
    # Check MongoDB connection
    print("\n   🗄️  MONGODB CHECK:")
    try:
        import pymongo
        from dotenv import load_dotenv
        load_dotenv()
        
        uri = os.getenv("MONGODB_URI")
        if uri:
            print("   ✓ MONGODB_URI found in .env")
            try:
                client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=3000)
                client.server_info()
                print("   ✓ MongoDB connection successful")
                
                # Check collections
                db_name = os.getenv("MONGODB_DATABASE", "aqi_predictor")
                db = client[db_name]
                collections = db.list_collection_names()
                print(f"   📊 Collections: {len(collections)} found")
                
                # Check key collections
                key_collections = ["aqi_measurements", "aqi_features", "model_registry"]
                for coll in key_collections:
                    if coll in collections:
                        count = db[coll].count_documents({})
                        print(f"      ✓ {coll}: {count} records")
                    else:
                        print(f"      ⚠️ {coll}: not found")
                
                client.close()
            except Exception as e:
                print(f"   ❌ MongoDB connection failed: {e}")
        else:
            print("   ❌ MONGODB_URI not found in .env")
    except ImportError:
        print("   ❌ pymongo not installed")
    
    print(f"\n   📦 DEPENDENCIES:")
    required_packages = ["streamlit", "pymongo", "pandas", "numpy", "plotly", "scikit-learn"]
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"      ✓ {package}")
        except ImportError:
            print(f"      ❌ {package} (not installed)")
            all_good = False
    
    return all_good

def setup_environment():
    """Setup environment and install dependencies"""
    print(f"\n[{datetime.now()}] ⚙️ Setting up environment...")
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        print("   Creating basic requirements.txt...")
        with open("requirements.txt", "w") as f:
            f.write("""streamlit==1.28.0
pymongo==4.5.0
pandas==2.1.0
numpy==1.24.0
plotly==5.17.0
scikit-learn==1.3.0
python-dotenv==1.0.0
schedule==1.2.0
openmeteo-requests==1.0.0
requests-cache==1.1.0
retry-requests==0.1.0
""")
        print("   ✓ Created requirements.txt")
    
    # Install dependencies
    print("   Installing dependencies from requirements.txt...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("   ✓ Dependencies installed")
    except Exception as e:
        print(f"   ❌ Failed to install dependencies: {e}")
        print("   You can install manually: pip install -r requirements.txt")

def show_project_info():
    """Show project information"""
    print(f"\n[{datetime.now()}] ℹ️  Project Information")
    print("   " + "=" * 50)
    print("   PROJECT: AQI Karachi Prediction System")
    print("   AUTHOR: Ajiya Anwar")
    print("   " + "=" * 50)
    print("\n   📁 Your Project Structure:")
    print("""
    aqi-karachi/
    ├── data_pipeline/
    │   ├── collect_historical.py
    │   └── features.py
    ├── model_training/
    │   ├── train_models.py
    │   ├── train_time_series_models.py
    │   └── runallmodels.py
    ├── dashboard/
    │   └── dashboard.py
    ├── cicd/
    │   └── pipeline.py
    ├── notebook/
    │   └── eda.py
    ├── data/ (auto-created)
    ├── logs/ (auto-created)
    ├── models/ (auto-created)
    ├── results/ (auto-created)
    ├── .env (your credentials)
    ├── requirements.txt
    ├── testdata.py
    └── run.py (this file)
    """)

def full_pipeline():
    """Run the complete pipeline"""
    print(f"\n[{datetime.now()}] 🚀 Starting FULL PIPELINE...")
    print("   This will run all steps sequentially:")
    print("   1. Data collection")
    print("   2. Feature engineering")
    print("   3. Model training (all models)")
    print("   4. EDA analysis")
    print("   5. Dashboard launch")
    print("   " + "-" * 50)
    
    response = input("\n   Continue? (y/n): ").strip().lower()
    if response != 'y':
        print("   Cancelled.")
        return
    
    steps = [
        ("📥 Collecting data", run_collect_data),
        ("🔧 Creating features", run_feature_engineering),
        ("🤖 Training models", run_all_models),
        ("📈 Running EDA", run_eda),
        ("📊 Launching dashboard", run_dashboard)
    ]
    
    for step_name, step_func in steps:
        print(f"\n[{datetime.now()}] {step_name}...")
        try:
            step_func()
        except KeyboardInterrupt:
            print(f"\n[{datetime.now()}] ⏹️  Step interrupted by user")
            continue
        except Exception as e:
            print(f"\n[{datetime.now()}] ❌ Error in {step_name}: {e}")
            continue

def main():
    """Main menu - Updated for your structure"""
    while True:
        print_header()
        
        print("\n📋 MAIN MENU (Updated for Your Structure):")
        print("=" * 60)
        print("DATA PIPELINE:")
        print("1. 📥 Collect 45 days historical data")
        print("2. 🔧 Create features from raw data")
        print("")
        print("MODEL TRAINING:")
        print("3. 🤖 Train ML models (train_models.py)")
        print("4. 📈 Run Time Series models (train_time_series_models.py)")
        print("5. 🚀 Run ALL models (runallmodels.py)")
        print("")
        print("ANALYSIS & VISUALIZATION:")
        print("6. 📊 Launch Streamlit dashboard")
        print("7. 📈 Run EDA analysis")
        print("8. 🧪 Run tests")
        print("")
        print("SYSTEM & MAINTENANCE:")
        print("9. 🔄 Run CI/CD pipeline")
        print("A. 🔍 Check system status")
        print("B. ⚙️ Setup environment")
        print("C. ℹ️ Show project info")
        print("D. 🚀 Run FULL PIPELINE")
        print("")
        print("0. 🚪 Exit")
        print("-" * 60)
        
        choice = input("\nSelect option (0-9, A-D): ").strip().upper()
        
        if choice == "1":
            run_collect_data()
        elif choice == "2":
            run_feature_engineering()
        elif choice == "3":
            run_ml_training()
        elif choice == "4":
            run_time_series()
        elif choice == "5":
            run_all_models()
        elif choice == "6":
            run_dashboard()
        elif choice == "7":
            run_eda()
        elif choice == "8":
            run_tests()
        elif choice == "9":
            run_cicd()
        elif choice == "A":
            check_system()
        elif choice == "B":
            setup_environment()
        elif choice == "C":
            show_project_info()
        elif choice == "D":
            full_pipeline()
        elif choice == "0":
            print("\n👋 Goodbye! Stay safe from pollution! 🌫️")
            break
        else:
            print("\n❌ Invalid choice. Please select 0-9 or A-D.")
        
        if choice != "0":
            input("\nPress Enter to continue...")
            os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == "__main__":
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    main()