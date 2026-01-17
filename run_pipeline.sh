#!/bin/bash
# Complete pipeline runner

echo "🔧 Running complete AQI ML pipeline..."

# Step 1: Data collection
echo "📥 Step 1: Collecting data..."
python save_aqi_data.py --once

# Step 2: Feature engineering
echo "🔧 Step 2: Feature engineering..."
python feature_engineering.py
# Step 3: Train traditional models
echo "🤖 Step 3: Training traditional ML models..."
python train_models.py

# Step 4: Train time-series models
echo "⏰ Step 4: Training time-series models..."
python train_time_series_models.py

# Step 5: Check results
echo "📊 Step 5: Checking results..."
python check_training_results.py
echo "✅ Pipeline complete!"
echo "🚀 Launch dashboard with: streamlit run app/comparison_dashboard.py"
