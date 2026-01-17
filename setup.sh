#!/bin/bash
# Setup script for AQI Prediction System

echo "🚀 Setting up AQI Prediction System..."

# Check Python version
python3 --version

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
# Create environment file
echo "⚙️  Creating environment configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️  Please edit .env file with your MongoDB credentials"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data models results logs
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your MongoDB URI"
echo "2. Run: python save_aqi_data.py (for data collection)"
echo "3. Run: python feature_engineering.py (for feature creation)"
echo "4. Run: python train_models.py (to train models)"
echo "5. Run: streamlit run app/comparison_dashboard.py (for dashboard)"

