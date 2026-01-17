"""
Streamlit Dashboard for Open-Meteo AQI Data with MongoDB Integration
"""
# Add these imports at the top
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Add after imports
def get_mongodb_connection():
    """Connect to MongoDB Atlas"""
    try:
        client = MongoClient(os.getenv("MONGODB_URI"))
        db = client[os.getenv("MONGODB_DATABASE")]
        return db
    except Exception as e:
        st.warning(f"MongoDB connection failed: {e}")
        return None

def get_historical_data(days=7):
    """Fetch historical AQI data from MongoDB"""
    db = get_mongodb_connection()
    if db and "aqi_measurements" in db.list_collection_names():
        collection = db["aqi_measurements"]
        
        # Get data from last N days
        cutoff_date = datetime.now() - timedelta(days=days)
        
        cursor = collection.find({
            "timestamp": {"$gte": cutoff_date.isoformat()}
        }).sort("timestamp", 1)
        
        data = list(cursor)
        if data:
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
    
    return pd.DataFrame()

# Add in main() function after current data fetch:
def main():
    # ... existing code ...
    
    # Fetch additional data
    with st.spinner("Fetching live data from Open-Meteo..."):
        current_data = fetch_current_aqi()
        forecast_data = fetch_forecast()
        historical_data = get_historical_data(days=7)  # NEW
    
    # Add a MongoDB status indicator
    st.sidebar.markdown("## 🗄️ Database Status")
    db = get_mongodb_connection()
    if db:
        st.sidebar.success("✅ MongoDB Connected")
        
        # Show collection stats
        collections = db.list_collection_names()
        st.sidebar.info(f"Collections: {len(collections)}")
        
        if "aqi_measurements" in collections:
            count = db["aqi_measurements"].count_documents({})
            st.sidebar.metric("Stored Measurements", f"{count:,}")
    else:
        st.sidebar.error("❌ MongoDB Not Connected")
    
    # Add historical trends section
    if not historical_data.empty:
        st.markdown("---")
        st.markdown("## 📈 Historical Trends (From MongoDB)")
        
        fig = px.line(
            historical_data,
            x="timestamp",
            y="aqi",
            title="AQI History (Last 7 Days)",
            markers=True
        )
        
        # Add AQI level backgrounds
        fig.update_layout(
            shapes=[
                dict(type="rect", xref="paper", yref="y",
                     x0=0, x1=1, y0=0, y1=50,
                     fillcolor="green", opacity=0.1, line_width=0),
                dict(type="rect", xref="paper", yref="y",
                     x0=0, x1=1, y0=51, y1=100,
                     fillcolor="yellow", opacity=0.1, line_width=0),
                dict(type="rect", xref="paper", yref="y",
                     x0=0, x1=1, y0=101, y1=150,
                     fillcolor="orange", opacity=0.1, line_width=0),
                dict(type="rect", xref="paper", yref="y",
                     x0=0, x1=1, y0=151, y1=500,
                     fillcolor="red", opacity=0.1, line_width=0)
            ]
        )
        
        st.plotly_chart(fig, use_container_width=True)