# 🌫️ AQI Karachi - Air Quality Prediction System

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green?logo=mongodb)
![GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub_Actions-black?logo=github)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit)
![ML](https://img.shields.io/badge/ML-Pipeline-orange)
![Status](https://img.shields.io/badge/Status-Production_Ready-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A comprehensive machine learning pipeline for predicting Air Quality Index (AQI) in Karachi, Pakistan with automated data collection, feature engineering, model training, and real-time dashboard visualization.

🔗 **Live Dashboard:** (https://aqi-karachi-tfgqthvmbxhirrtwrtuxnd.streamlit.app/)

## 🎯 Project Overview

This system provides **3-day AQI forecasts** for Karachi using a hybrid ML approach combining:
- **Machine Learning models** (Random Forest, XGBoost, Neural Networks)
- **Time Series forecasting** (Prophet, SARIMA)
- **Ensemble methods** for improved accuracy

## 🚀 Key Features

### 📊 **Automated Data Pipeline**
- **Real-time data collection** every 3 hours from multiple sources
- **Automated feature engineering** with rolling statistics
- **Continuous model training** and evaluation
- **Database persistence** with MongoDB Atlas

### 🤖 **Intelligent Forecasting**
- **Multi-model approach** (3 ML models + 2 time series models)
- **Ensemble predictions** for robust forecasts
- **Model versioning** and performance tracking
- **Automatic retraining** based on performance thresholds

### ⚡ **Production-Ready Infrastructure**
- **Fully automated CI/CD** with GitHub Actions
- **Scheduled executions** (data collection, training, predictions)
- **Production monitoring** and alerting
- **Scalable cloud architecture**

### 🌐 **Interactive Dashboard**
- **Real-time AQI visualization**
- **3-day forecast display**
- **Historical trend analysis**
- **Model performance metrics**
- **Geospatial mapping**

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   ML Pipeline   │    │   Dashboard     │
│  • Open-Meteo   │────▶ • Data Collect  │────▶ • Streamlit App │
│  • Weather APIs │    │ • Feature Eng   │    │ • Real-time Viz │
│  • Historical   │    │ • Model Train   │    │ • Forecasts     │
└─────────────────┘    │ • Predictions   │    └─────────────────┘
                       └─────────────────┘
                               │
                       ┌─────────────────┐
                       │   MongoDB Atlas │
                       │  • AQI Data     │
                       │  • Models       │
                       │  • Forecasts    │
                       └─────────────────┘
```

## 📅 Automated Schedule

The system runs **24/7** with the following automated schedule (UTC):

| Time (UTC) | Frequency | Task | Karachi Time |
|------------|-----------|------|--------------|
| **00:00, 03:00, 06:00, 09:00, 12:00, 15:00, 18:00, 21:00** | Every 3 hours | Data Collection | 05:00, 08:00, 11:00, 14:00, 17:00, 20:00, 23:00, 02:00 |
| **01:00** | Daily | Feature Engineering | 06:00 |
| **04:00 & 16:00** | Daily (2x) | Model Training | 09:00 & 21:00 |
| **05:00** | Sundays | Full Pipeline Run | 10:00 |

## 🛠️ Tech Stack

### **Backend & ML**
- **Python 3.10+** - Core programming language
- **Scikit-learn, XGBoost** - Machine learning models
- **Prophet, SARIMA** - Time series forecasting
- **Pandas, NumPy** - Data processing

### **Database & Storage**
- **MongoDB Atlas** - Cloud database for AQI data and models
- **MongoDB Collections:**
  - `aqi_predictor` - Raw and processed AQI data
  - `aqi_feature_store` - Engineered features
  - `aqi_model_registry` - Model versions and metadata
  - `ml_forecasts_3day` - ML model predictions
  - `timeseries_forecasts_3day` - Time series predictions
  - `ensemble_forecasts_3day` - Ensemble predictions

### **Infrastructure & DevOps**
- **GitHub Actions** - CI/CD automation
- **Ubuntu Latest** - Execution environment
- **Environment Variables** - Secure configuration management

### **Frontend & Visualization**
- **Streamlit** - Interactive dashboard
- **Plotly** - Advanced visualizations
- **Geospatial mapping** - Location-based displays

## 📦 Installation & Setup

### **Local Development**
```bash
# Clone repository
git clone https://github.com/AjiyaAnwar/aqi-Karachi.git
cd aqi-karachi

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your MongoDB URI and API keys

# Run pipeline locally
python data_pipeline/collect_historical.py
python data_pipeline/feature.py
python model_training/combinedtraining.py

# Launch dashboard
streamlit run dashboard/app.py
```

### **Environment Variables**
Create a `.env` file with:
```env
# MongoDB Configuration
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_DATABASE=aqi_predictor
MODEL_REGISTRY_DB=aqi_model_registry

# City Configuration
CITY_NAME=Karachi
CITY_LAT=24.8607
CITY_LON=67.0011
TIMEZONE=Asia/Karachi

# API Keys (if using external APIs)
OPENWEATHER_API_KEY=your_key_here
```

## 🔄 CI/CD Pipeline

The system uses **GitHub Actions** for automated workflow:

```yaml
# Key Pipeline Features:
# 1. Automated testing on every push
# 2. Scheduled execution (see schedule above)
# 3. MongoDB connection validation
# 4. Model training and deployment
# 5. Performance verification
```

## 📊 Data Flow

1. **Data Collection** → Fetch weather and pollution data
2. **Feature Engineering** → Create predictive features
3. **Model Training** → Train ML and time series models
4. **Predictions** → Generate 3-day AQI forecasts
5. **Storage** → Save forecasts to MongoDB
6. **Visualization** → Display in Streamlit dashboard

## 🎯 Model Architecture

### **Ensemble Approach:**
- **Base Models:** Random Forest, XGBoost, Neural Network, Prophet, SARIMA
- **Ensemble Method:** Weighted average based on recent performance
- **Validation:** Cross-validation with temporal splits
- **Metrics:** MAE, RMSE, R² Score

### **Key Features:**
- Meteorological data (temperature, humidity, wind)
- Temporal features (hour, day, month, season)
- Lag features (previous hour/day AQI)
- Rolling statistics (7-day averages)

## 📈 Performance Metrics

- **Prediction Horizon:** 3 days (72 hours)
- **Update Frequency:** Hourly forecasts
- **Accuracy Target:** < 15% MAE improvement over baseline
- **Model Retraining:** When performance drops below threshold

## 👩‍💻 Author

**Ajiya Anwar**  
🎓 Data Science Undergraduate  
📧 [LinkedIn Profile](https://www.linkedin.com/in/ajiya-anwar)  
🐙 [GitHub Profile](https://github.com/AjiyaAnwar)


## 🙏 Acknowledgments

- Open-Meteo for weather data API
- MongoDB Atlas for cloud database
- Streamlit for dashboard framework
- GitHub for CI/CD infrastructure

---

**🌟 Star this repo if you find it useful!**  
**🐛 Report issues in the GitHub Issues section**  
**💡 Contributions and suggestions are welcome!**

--- 
*Project Status: Production - Active Monitoring*
