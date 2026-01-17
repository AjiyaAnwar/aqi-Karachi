"""
Check model training results
"""
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

client = MongoClient(os.getenv('MONGODB_URI'))
db = client[os.getenv('MONGODB_DATABASE')]

print("📊 MODEL TRAINING RESULTS")
print("=" * 50)

# Check models in registry
models_count = db.model_registry.count_documents({})
print(f"\n🤖 Models in registry: {models_count}")

if models_count > 0:
    # Get latest model
    latest_model = db.model_registry.find_one(
        sort=[('created_at', -1)]
    )
    
    print(f"\n🎯 LATEST MODEL:")
    print(f"  ID: {latest_model.get('model_id')}")
    print(f"  Name: {latest_model.get('model_name')}")
    print(f"  Type: {latest_model.get('model_type')}")
    print(f"  Created: {latest_model.get('created_at')}")
    
    metrics = latest_model.get('metrics', {})
    print(f"\n📈 PERFORMANCE METRICS:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    features = latest_model.get('features_used', [])
    print(f"\n🔧 FEATURES USED ({len(features)}):")
    for i, feat in enumerate(sorted(features), 1):
        print(f"  {i:2d}. {feat}")

# Check predictions
predictions_count = db.aqi_predictions.count_documents({})
print(f"\n🔮 PREDICTIONS: {predictions_count}")

if predictions_count > 0:
    predictions = list(db.aqi_predictions.find({}, {'_id': 0}))
    df_pred = pd.DataFrame(predictions)
    
    print(f"\n📅 3-DAY FORECAST:")
    for _, row in df_pred.iterrows():
        date = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
        aqi = row.get('predicted_aqi', 0)
        print(f"  {date}: AQI {aqi:.1f}")

# Load CSV results if they exist
print(f"\n💾 LOCAL FILES:")
try:
    if os.path.exists('results/model_comparison.csv'):
        df_results = pd.read_csv('results/model_comparison.csv')
        print(f"  results/model_comparison.csv - {len(df_results)} models compared")
        
        # Show top 3 models
        print(f"\n🏆 TOP 3 MODELS:")
        top_models = df_results.nlargest(3, 'Test R²')
        for i, (_, row) in enumerate(top_models.iterrows(), 1):
            print(f"  {i}. {row['Model']} - R²: {row['Test R²']:.3f}, MAE: {row['Test MAE']:.2f}")
except:
    print("  No local results found")

client.close()
