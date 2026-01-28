    # check_models.py
import os
import pickle
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def check_and_fix_model_registry():
    """Check if models are properly registered in MongoDB"""
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('MONGODB_DATABASE')]
    
    print("🔍 Checking model_registry collection...")
    
    # Check model_registry
    models = list(db.model_registry.find({}))
    
    if not models:
        print("❌ No models found in model_registry")
        return False
    
    print(f"✅ Found {len(models)} models in model_registry")
    
    for i, model in enumerate(models, 1):
        print(f"\n📊 Model #{i}:")
        print(f"   Name: {model.get('model_name', 'N/A')}")
        print(f"   Type: {model.get('model_type', 'N/A')}")
        print(f"   Created: {model.get('created_at', 'N/A')}")
        print(f"   Production: {model.get('is_production', False)}")
        
        metrics = model.get('metrics', {})
        if metrics:
            print(f"   Metrics: R²={metrics.get('Test R²', 0):.3f}, MAE={metrics.get('Test MAE', 0):.2f}")
        else:
            print("   ⚠️ No metrics found")
    
    client.close()
    return True

def create_dummy_model_if_empty():
    """Create a dummy model if no models exist (for testing)"""
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('MONGODB_DATABASE')]
    
    models = list(db.model_registry.find({}))
    
    if not models:
        print("\n📝 Creating dummy model for testing...")
        
        dummy_model = {
            'model_id': 'dummy_model_20240123',
            'model_name': 'Random Forest',
            'model_type': 'RandomForestRegressor',
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'metrics': {
                'Test R²': 0.85,
                'Test MAE': 12.5,
                'Test RMSE': 18.3
            },
            'features_used': ['aqi', 'pm25', 'pm10', 'hour', 'day_of_week'],
            'model_bytes': b'dummy',
            'scaler_bytes': b'dummy',
            'is_production': True,
            'description': 'Dummy model for testing dashboard'
        }
        
        db.model_registry.insert_one(dummy_model)
        print("✅ Created dummy model")
    
    client.close()

if __name__ == "__main__":
    print("🔧 Model Registry Check & Fix")
    print("=" * 50)
    
    if not check_and_fix_model_registry():
        create_dummy_model_if_empty()
    
    print("\n✅ Done! Refresh your dashboard to see model metrics.")
