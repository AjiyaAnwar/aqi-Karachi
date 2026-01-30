"""
COMPLETE FIX: Clear corrupted data and verify everything
"""
import subprocess
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

def complete_fix():
    print("=" * 70)
    print("🚀 COMPLETE FIX FOR AQI PREDICTION SYSTEM")
    print("=" * 70)
    
    client = MongoClient(os.getenv('MONGODB_URI'))
    
    # Step 1: Clear corrupted data
    print("\n1️⃣ CLEARING CORRUPTED DATA")
    print("-" * 40)
    
    main_db = client['aqi_predictor']
    if 'aqi_features' in main_db.list_collection_names():
        count = main_db['aqi_features'].count_documents({})
        main_db['aqi_features'].delete_many({})
        print(f"🗑️  Cleared {count} corrupted records from main DB")
    
    # Also clear any other corrupted collections
    corrupted_collections = ['features', 'processed_features', 'ml_features']
    for coll in corrupted_collections:
        if coll in main_db.list_collection_names():
            count = main_db[coll].count_documents({})
            main_db[coll].delete_many({})
            print(f"🗑️  Cleared {count} records from {coll}")
    
    # Step 2: Verify Feature Store
    print("\n2️⃣ VERIFYING FEATURE STORE")
    print("-" * 40)
    
    fs_db = client['aqi_feature_store']
    if 'aqi_features' in fs_db.list_collection_names():
        count = fs_db['aqi_features'].count_documents({})
        sample = fs_db['aqi_features'].find_one({}, {'_id': 0})
        
        print(f"📊 Feature Store has {count} records")
        print(f"✅ Contains engineered features: {'target_24h' in sample}")
        print(f"✅ Contains lag features: {any('lag_' in k for k in sample.keys())}")
    else:
        print("❌ No features in Feature Store - running feature engineering...")
        subprocess.run(['python', 'data_pipeline/features.py'])
    
    # Step 3: Check correlation
    print("\n3️⃣ CHECKING TARGET CORRELATION")
    print("-" * 40)
    
    if 'aqi_features' in fs_db.list_collection_names():
        import pandas as pd
        
        # Get sample data
        cursor = list(fs_db['aqi_features'].find(
            {}, 
            {'_id': 0, 'aqi': 1, 'target_24h': 1, 'target_72h': 1}
        ).limit(100))
        
        if cursor:
            df = pd.DataFrame(cursor)
            corr_24h = df['aqi'].corr(df['target_24h'])
            corr_72h = df['aqi'].corr(df['target_72h'])
            
            print(f"📊 Correlation aqi vs target_24h: {corr_24h:.4f}")
            print(f"📊 Correlation aqi vs target_72h: {corr_72h:.4f}")
            
            if corr_24h > 0.5:
                print("✅ target_24h correlation is GOOD (should get positive R²)")
            else:
                print("⚠️  target_24h correlation is LOW (might still get negative R²)")
                print("💡 Try shorter prediction horizon (3h or 6h)")
    
    client.close()
    
    # Step 4: Test training
    print("\n4️⃣ TESTING TRAINING")
    print("-" * 40)
    
    print("Running train_models.py...")
    result = subprocess.run(
        ['python', 'model_training/train_models.py'],
        capture_output=True,
        text=True
    )
    
    # Check output
    if result.returncode == 0:
        print("✅ Training completed successfully")
        
        # Look for R² in output
        lines = result.stdout.split('\n')
        for line in lines:
            if 'R²' in line or 'Test R²' in line:
                print(f"📊 {line.strip()}")
    else:
        print("❌ Training failed")
        print(f"Error: {result.stderr[:200]}")
    
    print("\n" + "=" * 70)
    print("🎯 NEXT STEPS:")
    print("=" * 70)
    print("1. If R² is still negative, try shorter prediction (3h)")
    print("2. Check your train_models.py is reading from Feature Store")
    print("3. Run full pipeline: python model_training/runallmodels.py")
    print("4. Check dashboard for forecasts")

if __name__ == "__main__":
    complete_fix()