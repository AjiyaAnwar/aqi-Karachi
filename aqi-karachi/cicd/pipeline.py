# cicd/pipeline.py (Updated)
"""CI/CD Pipeline for AQI Karachi - Enhanced with MongoDB support"""
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
    def __init__(self, use_mongodb=False):
        self.project_root = PROJECT_ROOT
        self.data_available = False
        self.use_mongodb = use_mongodb
        
        if use_mongodb:
            try:
                from .mongodb_utils import MongoDBManager
                mongodb_uri = os.getenv("MONGODB_URI")
                if mongodb_uri:
                    self.mongo_manager = MongoDBManager(mongodb_uri)
                    logger.info("✅ MongoDB integration enabled")
                else:
                    logger.warning("⚠️ MONGODB_URI not set, MongoDB integration disabled")
                    self.use_mongodb = False
            except ImportError:
                logger.warning("⚠️ MongoDB utils not available, running in local mode")
                self.use_mongodb = False
    
    def run_script(self, script_path, description=""):
        """Run a Python script with proper error handling"""
        # ... (keep your existing run_script method) ...
    
    def log_to_mongo(self, step, status, details=None):
        """Log execution to MongoDB if enabled"""
        if self.use_mongodb:
            log_id = self.mongo_manager.log_pipeline_step(
                step=step,
                status=status,
                details=details
            )
            return log_id
        return None
    
    def run_data_collection(self):
        """Run data collection scripts"""
        logger.info("📥 COLLECTING DATA...")
        
        # Log start
        log_id = self.log_to_mongo('data_collection', 'started')
        
        try:
            # Try historical data collection first
            success = self.run_script(
                "data_pipeline/collect_historical.py",
                "Historical data collection"
            )
            
            if success:
                self.data_available = self.check_data_exists()
                
                # Log success
                self.log_to_mongo('data_collection', 'completed', {
                    'data_available': self.data_available
                })
                
                if self.data_available:
                    logger.info("✅ Data collection successful, data is available")
                else:
                    logger.warning("⚠️ Data collection ran but no data files were created")
            else:
                # Log failure
                self.log_to_mongo('data_collection', 'failed')
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Data collection failed: {str(e)}")
            self.log_to_mongo('data_collection', 'error', {'error': str(e)})
            return False
    
    def run_model_training(self):
        """Run model training with MongoDB integration"""
        # Check if we have processed data
        if not self.check_data_exists():
            logger.warning("⚠️ Skipping model training - no data available")
            logger.info("💡 Running data pipeline first...")
            
            # Run data collection and feature engineering
            self.run_data_collection()
            self.run_feature_engineering()
        
        logger.info("🤖 TRAINING MODELS...")
        
        # Log start
        log_id = self.log_to_mongo('model_training', 'started')
        
        try:
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
            
            # If MongoDB is enabled, store models
            if success and self.use_mongodb:
                self._store_trained_models()
            
            # Log completion
            self.log_to_mongo('model_training', 'completed', {
                'success': success,
                'parent_log_id': log_id
            })
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {str(e)}")
            self.log_to_mongo('model_training', 'error', {
                'error': str(e),
                'parent_log_id': log_id
            })
            return False
    
    def _store_trained_models(self):
        """Store trained models in MongoDB"""
        try:
            # Import your training script
            sys.path.append(os.path.join(self.project_root, 'model_training'))
            from runallmodels import train_and_evaluate_models
            
            # Train and get models
            models = train_and_evaluate_models()
            
            # Store each model in MongoDB
            for model_name, model_data in models.items():
                model, metrics = model_data
                self.mongo_manager.store_model(
                    model=model,
                    metrics=metrics,
                    model_name=model_name,
                    feature_version='local_' + datetime.now().strftime('%Y%m%d_%H%M')
                )
            
            logger.info(f"💾 Stored {len(models)} models in MongoDB")
            
        except Exception as e:
            logger.error(f"❌ Failed to store models in MongoDB: {str(e)}")
    
    # ... (keep other methods like check_data_exists, run_feature_engineering, etc.) ...

def main():
    """Main pipeline orchestrator - enhanced for dual mode"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AQI Karachi CI/CD Pipeline')
    parser.add_argument('--use-mongodb', action='store_true', 
                       help='Enable MongoDB integration')
    parser.add_argument('--step', type=str, choices=['data', 'features', 'eda', 'train', 'full'],
                       help='Run specific step only')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("🚀 AQI KARACHI - ENHANCED CI/CD PIPELINE")
    logger.info(f"📊 MongoDB: {'ENABLED' if args.use_mongodb else 'DISABLED'}")
    logger.info("=" * 60)
    
    executor = PipelineExecutor(use_mongodb=args.use_mongodb)
    
    # If specific step requested
    if args.step:
        steps = {
            'data': executor.run_data_collection,
            'features': executor.run_feature_engineering,
            'eda': executor.run_eda,
            'train': executor.run_model_training,
            'full': executor.run_full_pipeline
        }
        steps[args.step]()
        return
    
    # Otherwise run scheduled tasks
    # ... (keep your existing schedule setup) ...

if __name__ == "__main__":
    main()