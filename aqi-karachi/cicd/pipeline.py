"""
CI/CD Pipeline for AQI Karachi
Fixed to handle task dependencies properly
"""
import schedule
import time
import subprocess
import sys
import os
from datetime import datetime
import logging

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
log_dir = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'pipeline.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PipelineExecutor:
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.data_available = False
        
    def run_script(self, script_path, description=""):
        """Run a Python script with proper error handling"""
        abs_path = os.path.join(self.project_root, script_path)
        
        if not os.path.exists(abs_path):
            logger.error(f"❌ Script not found: {abs_path}")
            return False
        
        logger.info(f"▶️  Running: {description or os.path.basename(script_path)}")
        try:
            result = subprocess.run(
                [sys.executable, abs_path],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {description} completed successfully")
                if result.stdout.strip():
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            logger.info(f"   {line.strip()}")
                return True
            else:
                logger.error(f"❌ {description} failed with code {result.returncode}")
                if result.stderr:
                    for line in result.stderr.strip().split('\n')[-10:]:  # Last 10 lines
                        if line.strip():
                            logger.error(f"   {line.strip()}")
                return False
        except Exception as e:
            logger.error(f"❌ Exception in {description}: {str(e)}")
            return False
    
    def check_data_exists(self):
        """Check if data exists before running EDA"""
        data_dirs = [
            os.path.join(self.project_root, 'data'),
            os.path.join(self.project_root, 'results'),
            os.path.join(self.project_root, 'data_pipeline')
        ]
        
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                for root, dirs, files in os.walk(data_dir):
                    if files:
                        # Check for data files (CSV, JSON, etc.)
                        data_files = [f for f in files if f.endswith(('.csv', '.json', '.pkl', '.parquet'))]
                        if data_files:
                            logger.info(f"📁 Found data files in {root}: {len(data_files)} files")
                            return True
        
        logger.warning("⚠️  No data files found")
        return False
    
    def run_data_collection(self):
        """Run data collection scripts"""
        logger.info("📥 COLLECTING DATA...")
        
        # Try historical data collection first
        success = self.run_script(
            "data_pipeline/collect_historical.py",
            "Historical data collection"
        )
        
        if success:
            self.data_available = self.check_data_exists()
            if self.data_available:
                logger.info("✅ Data collection successful, data is available")
            else:
                logger.warning("⚠️  Data collection ran but no data files were created")
        
        return success
    
    def run_feature_engineering(self):
        """Run feature engineering"""
        # Only run if data exists
        if not self.data_available and not self.check_data_exists():
            logger.warning("⚠️  Skipping feature engineering - no data available")
            return False
        
        logger.info("🔧 ENGINEERING FEATURES...")
        return self.run_script(
            "data_pipeline/features.py",
            "Feature engineering"
        )
    
    def run_eda(self):
        """Run exploratory data analysis"""
        # Only run if data exists
        if not self.data_available and not self.check_data_exists():
            logger.warning("⚠️  Skipping EDA - no data available")
            logger.info("💡 Running data collection first...")
            
            # Try to collect data
            if self.run_data_collection():
                self.data_available = True
            else:
                logger.error("❌ Cannot run EDA - data collection failed")
                return False
        
        logger.info("📊 RUNNING EXPLORATORY DATA ANALYSIS...")
        
        # First try the new intelligent EDA
        eda_paths = [
            "notebook/eda_intelligent.py",  # You should save the new script as this
            "notebook/eda.py"  # Fallback to original
        ]
        
        for eda_path in eda_paths:
            if os.path.exists(os.path.join(self.project_root, eda_path)):
                return self.run_script(
                    eda_path,
                    "Exploratory Data Analysis"
                )
        
        logger.error("❌ No EDA script found")
        return False
    
    def run_model_training(self):
        """Run model training"""
        # Check if we have processed data
        if not self.check_data_exists():
            logger.warning("⚠️  Skipping model training - no data available")
            logger.info("💡 Running data pipeline first...")
            
            # Run data collection and feature engineering
            self.run_data_collection()
            self.run_feature_engineering()
        
        logger.info("🤖 TRAINING MODELS...")
        
        # Try multiple training scripts
        training_scripts = [
            "model_training/runallmodels.py",
            "model_training/train_models.py",
            "model_training/train_time_series_models.py"
        ]
        
        success = True
        for script in training_scripts:
            if os.path.exists(os.path.join(self.project_root, script)):
                if not self.run_script(script, f"Model training: {os.path.basename(script)}"):
                    success = False
        
        return success
    
    def run_full_pipeline(self):
        """Run complete end-to-end pipeline"""
        logger.info("=" * 60)
        logger.info("🚀 STARTING FULL PIPELINE EXECUTION")
        logger.info("=" * 60)
        
        # Reset data flag
        self.data_available = False
        
        # Execute in order
        steps = [
            ("Data Collection", self.run_data_collection),
            ("Feature Engineering", self.run_feature_engineering),
            ("Exploratory Data Analysis", self.run_eda),
            ("Model Training", self.run_model_training)
        ]
        
        results = {}
        for step_name, step_func in steps:
            logger.info(f"\n{'='*40}")
            logger.info(f"STEP: {step_name}")
            logger.info(f"{'='*40}")
            results[step_name] = step_func()
            time.sleep(2)  # Brief pause between steps
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 60)
        
        all_success = True
        for step_name, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            logger.info(f"{step_name}: {status}")
            if not success:
                all_success = False
        
        if all_success:
            logger.info("\n🎉 ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
        else:
            logger.warning("\n⚠️  Some pipeline steps failed. Check logs for details.")
        
        logger.info("=" * 60)
        return all_success

def main():
    """Main pipeline orchestrator"""
    logger.info("=" * 60)
    logger.info("🚀 AQI KARACHI - CI/CD PIPELINE")
    logger.info("=" * 60)
    
    executor = PipelineExecutor()
    
    # Schedule tasks with proper dependencies
    # Daily at midnight: Full data collection
    schedule.every().day.at("00:00").do(executor.run_data_collection)
    
    # Daily at 1 AM: Feature engineering (depends on data)
    schedule.every().day.at("01:00").do(executor.run_feature_engineering)
    
    # Daily at 2 AM: EDA (depends on features)
    schedule.every().day.at("02:00").do(executor.run_eda)
    
    # Model training every 3 hours (will check for data)
    schedule.every(3).hours.do(executor.run_model_training)
    
    # Full pipeline every Sunday at 3 AM
    schedule.every().sunday.at("03:00").do(executor.run_full_pipeline)
    
    logger.info("\n📅 SCHEDULED TASKS:")
    logger.info("   00:00 - Data Collection")
    logger.info("   01:00 - Feature Engineering")
    logger.info("   02:00 - Exploratory Data Analysis")
    logger.info("   Every 3 hours - Model Training")
    logger.info("   Sunday 03:00 - Full Pipeline Run")
    
    logger.info("\n🔄 Running initial full pipeline...")
    
    # Run initial full pipeline
    executor.run_full_pipeline()
    
    logger.info("\n⏰ Pipeline scheduler running. Press Ctrl+C to stop.")
    logger.info("=" * 60)
    
    # Keep scheduler running
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("\n👋 Pipeline stopped by user.")
    except Exception as e:
        logger.error(f"❌ Pipeline crashed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()