"""
🎯 EDA ANALYSIS FOR AQI KARACHI
Comprehensive, adaptive EDA that finds and analyzes real data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pickle
import warnings
import sys
from datetime import datetime, timedelta
from pathlib import Path
import glob
import traceback
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class EDAAnalyzer:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results_dir = self.project_root / "results" / "eda_reports"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # AQI categories and colors
        self.aqi_categories = {
            (0, 50): ("Good", "#10B981", "Air quality is satisfactory"),
            (51, 100): ("Moderate", "#F59E0B", "Acceptable air quality"),
            (101, 150): ("Unhealthy for Sensitive", "#F97316", "Health effects for sensitive groups"),
            (151, 200): ("Unhealthy", "#EF4444", "Everyone may experience health effects"),
            (201, 300): ("Very Unhealthy", "#8B5CF6", "Health alert"),
            (301, 500): ("Hazardous", "#7C3AED", "Emergency conditions")
        }
        
        # Initialize logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        import logging
        self.logger = logging.getLogger('EDA_Analyzer')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.results_dir / 'eda_analysis.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def find_data_sources(self):
        """Intelligently find all possible data sources"""
        self.logger.info("🔍 Searching for data sources...")
        
        data_sources = []
        
        # Check common data directories
        directories_to_check = [
            self.project_root / "data",
            self.project_root / "data_pipeline",
            self.project_root / "results",
            self.project_root / "model_training",
            self.project_root
        ]
        
        for directory in directories_to_check:
            if directory.exists():
                # Look for CSV files
                csv_files = list(directory.glob("**/*.csv"))
                # Look for JSON files
                json_files = list(directory.glob("**/*.json"))
                # Look for pickle files
                pkl_files = list(directory.glob("**/*.pkl"))
                
                for file in csv_files + json_files + pkl_files:
                    # Skip temporary and cache files
                    if not any(x in str(file) for x in ['temp', 'cache', '__pycache__', '.git']):
                        try:
                            size_mb = file.stat().st_size / (1024 * 1024)
                            data_sources.append({
                                'path': file,
                                'type': file.suffix,
                                'size_mb': round(size_mb, 2)
                            })
                        except:
                            continue
        
        self.logger.info(f"📁 Found {len(data_sources)} potential data sources")
        return sorted(data_sources, key=lambda x: x['size_mb'], reverse=True)

    def load_data_intelligently(self):
        """Intelligently load the most relevant dataset"""
        data_sources = self.find_data_sources()
        
        if not data_sources:
            self.logger.error("❌ No data sources found!")
            return None
        
        self.logger.info(f"📊 Trying to load data from {len(data_sources)} sources...")
        
        # Try to find the most relevant file
        priority_patterns = [
            '*aqi*', '*pm2.5*', '*pollution*', '*air*quality*',
            '*training*', '*processed*', '*features*'
        ]
        
        for pattern in priority_patterns:
            for source in data_sources:
                if pattern.replace('*', '').lower() in str(source['path']).lower():
                    try:
                        df = self.load_file(source['path'])
                        if df is not None and not df.empty:
                            self.logger.info(f"✅ Successfully loaded: {source['path']}")
                            self.logger.info(f"   Shape: {df.shape}, Columns: {len(df.columns)}")
                            return df
                    except Exception as e:
                        self.logger.warning(f"⚠️  Failed to load {source['path']}: {e}")
        
        # If priority files failed, try any file
        for source in data_sources:
            if source['size_mb'] > 0.1:  # Skip very small files
                try:
                    df = self.load_file(source['path'])
                    if df is not None and not df.empty:
                        self.logger.info(f"✅ Loaded fallback file: {source['path']}")
                        return df
                except Exception as e:
                    continue
        
        return None

    def load_file(self, file_path):
        """Load file based on extension"""
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path, low_memory=False)
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    df = pd.DataFrame([data])
                else:
                    df = pd.DataFrame(data)
            elif file_path.suffix == '.pkl':
                with open(file_path, 'rb') as f:
                    df = pickle.load(f)
            else:
                return None
            
            # Basic cleaning
            if isinstance(df, pd.DataFrame):
                # Remove completely empty columns
                df = df.dropna(axis=1, how='all')
                # Convert date columns if present
                date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                for col in date_columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass
                
                self.logger.info(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
                return df
        except Exception as e:
            self.logger.error(f"❌ Error loading {file_path}: {e}")
            return None

    def analyze_data_structure(self, df):
        """Comprehensive data structure analysis"""
        self.logger.info("📐 Analyzing data structure...")
        
        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            'unique_counts': {col: df[col].nunique() for col in df.columns},
            'date_columns': [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])],
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
        }
        
        return analysis

    def create_comprehensive_summary(self, df):
        """Create comprehensive data summary"""
        summary = {
            'overall': {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'start_date': None,
                'end_date': None,
                'date_range_days': None
            },
            'missing_values': {},
            'statistical_summary': {},
            'aqi_analysis': {}
        }
        
        # Check for date columns
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        if date_cols:
            dates = pd.concat([df[col] for col in date_cols], ignore_index=True)
            dates = dates.dropna()
            if len(dates) > 0:
                summary['overall']['start_date'] = dates.min()
                summary['overall']['end_date'] = dates.max()
                summary['overall']['date_range_days'] = (dates.max() - dates.min()).days
        
        # Missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        summary['missing_values'] = pd.DataFrame({
            'missing_count': missing,
            'missing_percentage': missing_pct
        }).to_dict()
        
        # Statistical summary for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_df = df[numeric_cols].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
            summary['statistical_summary'] = stats_df.to_dict()
        
        # AQI specific analysis
        aqi_cols = [col for col in df.columns if 'aqi' in col.lower()]
        if aqi_cols:
            aqi_data = df[aqi_cols[0]].dropna()
            if len(aqi_data) > 0:
                summary['aqi_analysis'] = {
                    'mean': float(aqi_data.mean()),
                    'median': float(aqi_data.median()),
                    'std': float(aqi_data.std()),
                    'min': float(aqi_data.min()),
                    'max': float(aqi_data.max()),
                    'skewness': float(aqi_data.skew()),
                    'kurtosis': float(aqi_data.kurtosis()),
                    'percentile_25': float(aqi_data.quantile(0.25)),
                    'percentile_75': float(aqi_data.quantile(0.75)),
                    'category_distribution': self.categorize_aqi(aqi_data)
                }
        
        return summary

    def categorize_aqi(self, aqi_series):
        """Categorize AQI values"""
        categories = {}
        for (low, high), (name, color, desc) in self.aqi_categories.items():
            count = ((aqi_series >= low) & (aqi_series <= high)).sum()
            if count > 0:
                categories[name] = {
                    'count': int(count),
                    'percentage': round((count / len(aqi_series)) * 100, 2),
                    'color': color,
                    'description': desc
                }
        return categories

    def create_visual_analytics(self, df):
        """Create comprehensive visual analytics"""
        self.logger.info("🎨 Creating visual analytics...")
        
        # Create subdirectory for plots
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 1. AQI Distribution Analysis
        self.create_aqi_distribution_plots(df, plots_dir)
        
        # 2. Temporal Analysis
        self.create_temporal_analysis(df, plots_dir)
        
        # 3. Correlation Analysis
        self.create_correlation_analysis(df, plots_dir)
        
        # 4. Feature Importance Analysis
        self.create_feature_analysis(df, plots_dir)
        
        # 5. Interactive Dashboard
        self.create_interactive_dashboard(df, plots_dir)
        
        self.logger.info(f"✅ Visual analytics saved to {plots_dir}")

    def create_aqi_distribution_plots(self, df, plots_dir):
        """Create AQI distribution plots"""
        aqi_cols = [col for col in df.columns if 'aqi' in col.lower()]
        if not aqi_cols:
            return
        
        aqi_col = aqi_cols[0]
        aqi_data = df[aqi_col].dropna()
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Histogram with KDE
        axes[0, 0].hist(aqi_data, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        aqi_data.plot(kind='kde', ax=axes[0, 0], color='darkblue', linewidth=2)
        axes[0, 0].set_title('AQI Distribution with KDE', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('AQI')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Box plot with outliers
        axes[0, 1].boxplot(aqi_data, vert=False)
        axes[0, 1].set_title('AQI Box Plot', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('AQI')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Violin plot
        axes[0, 2].violinplot(aqi_data, showmeans=True, showmedians=True)
        axes[0, 2].set_title('AQI Violin Plot', fontsize=14, fontweight='bold')
        axes[0, 2].set_ylabel('AQI')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Q-Q Plot
        stats.probplot(aqi_data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normality Test)', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Cumulative Distribution
        sorted_aqi = np.sort(aqi_data)
        cdf = np.arange(1, len(sorted_aqi) + 1) / len(sorted_aqi)
        axes[1, 1].plot(sorted_aqi, cdf, linewidth=2, color='green')
        axes[1, 1].set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('AQI')
        axes[1, 1].set_ylabel('CDF')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. AQI Categories Pie Chart
        categories = self.categorize_aqi(aqi_data)
        if categories:
            labels = list(categories.keys())
            sizes = [categories[label]['percentage'] for label in labels]
            colors = [categories[label]['color'] for label in labels]
            axes[1, 2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                          startangle=90, textprops={'fontsize': 10})
            axes[1, 2].set_title('AQI Category Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'aqi_distribution_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save statistics
        stats_file = plots_dir / 'aqi_statistics.txt'
        with open(stats_file, 'w') as f:
            f.write("AQI Statistical Summary:\n")
            f.write("=" * 50 + "\n")
            f.write(f"Mean: {aqi_data.mean():.2f}\n")
            f.write(f"Median: {aqi_data.median():.2f}\n")
            f.write(f"Std Dev: {aqi_data.std():.2f}\n")
            f.write(f"Skewness: {aqi_data.skew():.2f}\n")
            f.write(f"Kurtosis: {aqi_data.kurtosis():.2f}\n")
            f.write(f"Minimum: {aqi_data.min():.2f}\n")
            f.write(f"Maximum: {aqi_data.max():.2f}\n")
            f.write(f"25th Percentile: {aqi_data.quantile(0.25):.2f}\n")
            f.write(f"75th Percentile: {aqi_data.quantile(0.75):.2f}\n")
            f.write("\nAQI Categories:\n")
            for category, info in categories.items():
                f.write(f"{category}: {info['percentage']}% ({info['count']} records)\n")

    def create_temporal_analysis(self, df, plots_dir):
        """Create temporal analysis plots"""
        # Find date column
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        aqi_cols = [col for col in df.columns if 'aqi' in col.lower()]
        
        if not date_cols or not aqi_cols:
            return
        
        date_col = date_cols[0]
        aqi_col = aqi_cols[0]
        
        # Create time series analysis
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        
        # 1. Time Series Plot
        axes[0, 0].plot(df[date_col], df[aqi_col], linewidth=1, alpha=0.7, color='blue')
        axes[0, 0].set_title('AQI Time Series', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('AQI')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Rolling Average (7-day)
        df_sorted = df.sort_values(date_col)
        rolling_avg = df_sorted.set_index(date_col)[aqi_col].rolling(window=7).mean()
        axes[0, 1].plot(rolling_avg.index, rolling_avg, linewidth=2, color='red')
        axes[0, 1].set_title('7-Day Rolling Average', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('AQI (7-day avg)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Hourly Pattern
        if 'hour' in df.columns or df[date_col].dt.hour.nunique() > 1:
            hour_col = 'hour' if 'hour' in df.columns else df[date_col].dt.hour
            hourly_avg = df.groupby(hour_col)[aqi_col].mean()
            axes[1, 0].bar(hourly_avg.index, hourly_avg.values, color='orange', alpha=0.7)
            axes[1, 0].set_title('Average AQI by Hour', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Hour of Day')
            axes[1, 0].set_ylabel('Average AQI')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Daily Pattern
        if 'day_of_week' in df.columns or df[date_col].dt.dayofweek.nunique() > 1:
            day_col = 'day_of_week' if 'day_of_week' in df.columns else df[date_col].dt.dayofweek
            daily_avg = df.groupby(day_col)[aqi_col].mean()
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            axes[1, 1].bar(range(len(daily_avg)), daily_avg.values, color='green', alpha=0.7)
            axes[1, 1].set_title('Average AQI by Day of Week', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Day of Week')
            axes[1, 1].set_ylabel('Average AQI')
            axes[1, 1].set_xticks(range(len(daily_avg)))
            axes[1, 1].set_xticklabels(day_names)
            axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Monthly Pattern
        if 'month' in df.columns or df[date_col].dt.month.nunique() > 1:
            month_col = 'month' if 'month' in df.columns else df[date_col].dt.month
            monthly_avg = df.groupby(month_col)[aqi_col].mean()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            axes[2, 0].bar(range(1, len(monthly_avg) + 1), monthly_avg.values, 
                          color='purple', alpha=0.7)
            axes[2, 0].set_title('Average AQI by Month', fontsize=14, fontweight='bold')
            axes[2, 0].set_xlabel('Month')
            axes[2, 0].set_ylabel('Average AQI')
            axes[2, 0].set_xticks(range(1, len(monthly_avg) + 1))
            axes[2, 0].set_xticklabels(month_names[:len(monthly_avg)])
            axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Seasonal Decomposition
        try:
            # Resample to daily if hourly data
            if len(df) > 100:  # Only if we have enough data
                ts = df_sorted.set_index(date_col)[aqi_col]
                if pd.infer_freq(ts.index) in ['H', 'h']:
                    ts = ts.resample('D').mean()
                
                if len(ts) > 30:
                    decomposition = seasonal_decompose(ts.dropna(), model='additive', period=7)
                    
                    axes[2, 1].plot(decomposition.trend.index, decomposition.trend, 
                                   label='Trend', linewidth=2, color='red')
                    axes[2, 1].plot(decomposition.seasonal.index, decomposition.seasonal, 
                                   label='Seasonal', linewidth=1, color='green', alpha=0.7)
                    axes[2, 1].plot(decomposition.resid.index, decomposition.resid, 
                                   label='Residual', linewidth=0.5, color='blue', alpha=0.5)
                    axes[2, 1].set_title('Seasonal Decomposition', fontsize=14, fontweight='bold')
                    axes[2, 1].set_xlabel('Date')
                    axes[2, 1].set_ylabel('AQI Components')
                    axes[2, 1].legend()
                    axes[2, 1].grid(True, alpha=0.3)
        except Exception as e:
            self.logger.warning(f"⚠️  Seasonal decomposition failed: {e}")
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'temporal_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

    def create_correlation_analysis(self, df, plots_dir):
        """Create correlation analysis"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return
        
        # Correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Correlation Heatmap
        im = axes[0, 0].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        axes[0, 0].set_title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
        axes[0, 0].set_xticks(range(len(numeric_cols)))
        axes[0, 0].set_yticks(range(len(numeric_cols)))
        axes[0, 0].set_xticklabels(numeric_cols, rotation=45, ha='right', fontsize=9)
        axes[0, 0].set_yticklabels(numeric_cols, fontsize=9)
        
        # Add correlation values
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                value = corr_matrix.iloc[i, j]
                color = 'white' if abs(value) > 0.5 else 'black'
                axes[0, 0].text(j, i, f'{value:.2f}', ha='center', va='center', 
                               color=color, fontsize=8, fontweight='bold')
        
        plt.colorbar(im, ax=axes[0, 0])
        
        # 2. Top Correlations with AQI
        aqi_cols = [col for col in numeric_cols if 'aqi' in col.lower()]
        if aqi_cols:
            aqi_corr = corr_matrix[aqi_cols[0]].drop(aqi_cols[0]).sort_values(ascending=False)
            top_10 = aqi_corr.head(10)
            
            colors = ['green' if x > 0 else 'red' for x in top_10.values]
            axes[0, 1].barh(range(len(top_10)), top_10.values, color=colors, alpha=0.7)
            axes[0, 1].set_yticks(range(len(top_10)))
            axes[0, 1].set_yticklabels(top_10.index)
            axes[0, 1].set_title(f'Top 10 Features Correlated with {aqi_cols[0]}', 
                               fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Correlation Coefficient')
            axes[0, 1].grid(True, alpha=0.3, axis='x')
            
            # Add correlation values on bars
            for i, v in enumerate(top_10.values):
                axes[0, 1].text(v, i, f' {v:.3f}', va='center', fontweight='bold')
        
        # 3. Scatter plot matrix for top correlated features
        if aqi_cols and len(numeric_cols) > 1:
            # Get top 3 correlated features
            top_correlated = aqi_corr.head(4).index.tolist()
            if len(top_correlated) >= 3:
                scatter_cols = [aqi_cols[0]] + top_correlated[:3]
                
                # Create scatter matrix
                from pandas.plotting import scatter_matrix
                scatter_df = df[scatter_cols].dropna()
                
                scatter_matrix(scatter_df, alpha=0.5, figsize=(12, 12), 
                              diagonal='hist', ax=axes[1, 0])
                axes[1, 0].set_title('Scatter Matrix of Top Correlated Features', 
                                    fontsize=14, fontweight='bold')
        
        # 4. Pairwise scatter plots
        if aqi_cols and len(numeric_cols) > 1:
            # Select 2-3 most correlated features
            if len(top_correlated) >= 2:
                feature1 = top_correlated[0]
                axes[1, 1].scatter(df[feature1], df[aqi_cols[0]], alpha=0.5, s=10)
                axes[1, 1].set_xlabel(feature1)
                axes[1, 1].set_ylabel(aqi_cols[0])
                axes[1, 1].set_title(f'{feature1} vs {aqi_cols[0]}', 
                                    fontsize=14, fontweight='bold')
                axes[1, 1].grid(True, alpha=0.3)
                
                # Add trend line
                try:
                    z = np.polyfit(df[feature1].dropna(), df[aqi_cols[0]].dropna(), 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(df[feature1].min(), df[feature1].max(), 100)
                    axes[1, 1].plot(x_range, p(x_range), "r--", linewidth=2, alpha=0.8)
                except:
                    pass
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'correlation_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save correlation matrix
        corr_matrix.to_csv(plots_dir / 'correlation_matrix.csv')

    def create_feature_analysis(self, df, plots_dir):
        """Create feature importance and analysis"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return
        
        # Feature importance using mutual information
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            # Find target (AQI)
            aqi_cols = [col for col in numeric_cols if 'aqi' in col.lower()]
            if not aqi_cols:
                return
            
            target_col = aqi_cols[0]
            feature_cols = [col for col in numeric_cols if col != target_col]
            
            # Prepare data
            X = df[feature_cols].fillna(df[feature_cols].median())
            y = df[target_col].fillna(df[target_col].median())
            
            # Calculate mutual information
            mi_scores = mutual_info_regression(X, y, random_state=42)
            mi_scores = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)
            
            # Create feature importance plot
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))
            
            # 1. Feature Importance Bar Chart
            top_n = min(15, len(mi_scores))
            axes[0, 0].barh(range(top_n), mi_scores.head(top_n).values, color='teal', alpha=0.7)
            axes[0, 0].set_yticks(range(top_n))
            axes[0, 0].set_yticklabels(mi_scores.head(top_n).index)
            axes[0, 0].set_title('Feature Importance (Mutual Information)', 
                               fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Mutual Information Score')
            axes[0, 0].invert_yaxis()
            axes[0, 0].grid(True, alpha=0.3, axis='x')
            
            # 2. Feature Distributions
            top_features = mi_scores.head(4).index.tolist()
            for i, feature in enumerate(top_features[:4]):
                row = i // 2
                col = i % 2
                axes[1, col].hist(df[feature].dropna(), bins=30, alpha=0.6, 
                                 color=plt.cm.tab10(i), edgecolor='black')
                axes[1, col].set_title(f'Distribution: {feature}', fontsize=12, fontweight='bold')
                axes[1, col].set_xlabel(feature)
                axes[1, col].set_ylabel('Frequency')
                axes[1, col].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'feature_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save feature importance scores
            mi_scores.to_csv(plots_dir / 'feature_importance_scores.csv')
            
        except Exception as e:
            self.logger.warning(f"⚠️  Feature importance analysis failed: {e}")

    def create_interactive_dashboard(self, df, plots_dir):
        """Create interactive HTML dashboard"""
        try:
            aqi_cols = [col for col in df.columns if 'aqi' in col.lower()]
            date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            
            if not aqi_cols or not date_cols:
                return
            
            # Create interactive time series plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df[date_cols[0]],
                y=df[aqi_cols[0]],
                mode='lines+markers',
                name='AQI',
                line=dict(color='blue', width=1),
                marker=dict(size=3),
                hovertemplate='<b>Date:</b> %{x}<br><b>AQI:</b> %{y:.1f}<extra></extra>'
            ))
            
            # Add rolling average
            df_sorted = df.sort_values(date_cols[0])
            rolling_avg = df_sorted.set_index(date_cols[0])[aqi_cols[0]].rolling(window=7).mean()
            
            fig.add_trace(go.Scatter(
                x=rolling_avg.index,
                y=rolling_avg.values,
                mode='lines',
                name='7-Day Rolling Avg',
                line=dict(color='red', width=2),
                hovertemplate='<b>Date:</b> %{x}<br><b>7-Day Avg AQI:</b> %{y:.1f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Interactive AQI Time Series Dashboard',
                xaxis_title='Date',
                yaxis_title='AQI',
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            
            # Save as HTML
            dashboard_path = plots_dir / 'interactive_dashboard.html'
            fig.write_html(str(dashboard_path))
            
            self.logger.info(f"✅ Interactive dashboard saved: {dashboard_path}")
            
        except Exception as e:
            self.logger.warning(f"⚠️  Interactive dashboard failed: {e}")

    def generate_insights_report(self, df, analysis, summary):
        """Generate comprehensive insights report"""
        self.logger.info("💡 Generating insights report...")
        
        report_path = self.results_dir / "eda_insights_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# 🎯 AQI Karachi - EDA Insights Report\n\n")
            f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            
            f.write("## 📊 Executive Summary\n\n")
            f.write(f"- **Total Records**: {summary['overall']['total_records']:,}\n")
            f.write(f"- **Total Features**: {summary['overall']['total_columns']}\n")
            
            if summary['overall']['date_range_days']:
                f.write(f"- **Time Period**: {summary['overall']['date_range_days']} days\n")
                f.write(f"- **From**: {summary['overall']['start_date']}\n")
                f.write(f"- **To**: {summary['overall']['end_date']}\n")
            
            f.write("\n## 📈 Key Findings\n\n")
            
            # AQI Analysis
            if summary.get('aqi_analysis'):
                aqi_stats = summary['aqi_analysis']
                f.write("### AQI Analysis\n\n")
                f.write(f"- **Average AQI**: {aqi_stats['mean']:.1f}\n")
                f.write(f"- **Median AQI**: {aqi_stats['median']:.1f}\n")
                f.write(f"- **AQI Range**: {aqi_stats['min']:.1f} to {aqi_stats['max']:.1f}\n")
                f.write(f"- **Standard Deviation**: {aqi_stats['std']:.1f}\n")
                
                # Add category insights
                if 'category_distribution' in aqi_stats:
                    f.write("\n### AQI Categories Distribution\n\n")
                    for category, info in aqi_stats['category_distribution'].items():
                        f.write(f"- **{category}**: {info['percentage']}% of records\n")
            
            # Data Quality
            missing_df = pd.DataFrame(summary['missing_values'])
            high_missing = missing_df[missing_df['missing_percentage'] > 20]
            if not high_missing.empty:
                f.write("\n### ⚠️ Data Quality Issues\n\n")
                f.write("Features with >20% missing values:\n\n")
                f.write(missing_df[['missing_percentage']].to_markdown())
                f.write("\n\n")
            
            # Temporal Insights
            if 'date_columns' in analysis and analysis['date_columns']:
                f.write("\n### 📅 Temporal Insights\n\n")
                f.write(f"- Date column found: {analysis['date_columns'][0]}\n")
                f.write(f"- Temporal granularity: Hourly/Daily data\n")
            
            # Feature Insights
            f.write("\n### 🔧 Feature Engineering Recommendations\n\n")
            f.write("1. **Handle Missing Values**:\n")
            f.write("   - Impute numeric features with median\n")
            f.write("   - Consider dropping features with >30% missing values\n\n")
            
            f.write("2. **Create Time-based Features**:\n")
            f.write("   - Hour of day, day of week, month, season\n")
            f.write("   - Rolling averages (7-day, 30-day)\n")
            f.write("   - Lag features (previous hour/day AQI)\n\n")
            
            f.write("3. **Create Interaction Features**:\n")
            f.write("   - Ratios between pollutant measurements\n")
            f.write("   - Combined pollution indices\n\n")
            
            f.write("### 🤖 Modeling Recommendations\n\n")
            f.write("1. **For Time Series Prediction**:\n")
            f.write("   - Use LSTM or Prophet models\n")
            f.write("   - Include lag features and rolling statistics\n")
            f.write("   - Consider exogenous variables (weather, traffic)\n\n")
            
            f.write("2. **For Classification (AQI Categories)**:\n")
            f.write("   - Random Forest or XGBoost\n")
            f.write("   - Handle class imbalance if present\n")
            f.write("   - Focus on feature importance from EDA\n\n")
            
            f.write("## 📁 Files Generated\n\n")
            f.write("- `eda_reports/plots/`: All visualization plots\n")
            f.write("- `eda_reports/eda_analysis.log`: Detailed execution log\n")
            f.write("- `eda_reports/eda_insights_report.md`: This report\n")
            f.write("- `eda_reports/plots/correlation_matrix.csv`: Correlation data\n")
            f.write("- `eda_reports/plots/feature_importance_scores.csv`: Feature importance\n")
            f.write("- `eda_reports/plots/interactive_dashboard.html`: Interactive dashboard\n\n")
            
            f.write("## 🚀 Next Steps\n\n")
            f.write("1. **Data Collection**: Ensure continuous data collection\n")
            f.write("2. **Feature Engineering**: Implement recommendations\n")
            f.write("3. **Model Training**: Train with identified important features\n")
            f.write("4. **Monitoring**: Set up alerts for data quality issues\n")
            f.write("5. **Deployment**: Deploy best-performing models\n")
        
        self.logger.info(f"✅ Insights report saved: {report_path}")
        return report_path

    def run(self):
        """Main execution method"""
        self.logger.info("=" * 80)
        self.logger.info("🎯 STARTING INTELLIGENT EDA ANALYSIS")
        self.logger.info("=" * 80)
        
        # Step 1: Find and load data
        df = self.load_data_intelligently()
        
        if df is None or df.empty:
            self.logger.error("❌ No valid data found for analysis!")
            self.logger.info("💡 Please ensure data collection scripts have run successfully.")
            self.logger.info("💡 Check if data exists in: data/, data_pipeline/, or results/ directories")
            return
        
        self.logger.info(f"✅ Successfully loaded dataset: {len(df)} records, {len(df.columns)} features")
        
        # Step 2: Analyze data structure
        analysis = self.analyze_data_structure(df)
        
        # Step 3: Create comprehensive summary
        summary = self.create_comprehensive_summary(df)
        
        # Step 4: Create visual analytics
        self.create_visual_analytics(df)
        
        # Step 5: Generate insights report
        report_path = self.generate_insights_report(df, analysis, summary)
        
        self.logger.info("=" * 80)
        self.logger.info("✅ EDA ANALYSIS COMPLETED SUCCESSFULLY!")
        self.logger.info("=" * 80)
        self.logger.info(f"📊 Data Analyzed: {len(df)} records, {len(df.columns)} features")
        self.logger.info(f"📁 Results saved to: {self.results_dir}")
        self.logger.info(f"📄 Insights report: {report_path}")
        self.logger.info("=" * 80)
        
        # Print key insights to console
        print("\n" + "=" * 80)
        print("🎯 KEY INSIGHTS SUMMARY")
        print("=" * 80)
        
        if summary.get('aqi_analysis'):
            aqi_stats = summary['aqi_analysis']
            print(f"\n📊 AQI Statistics:")
            print(f"   Average: {aqi_stats['mean']:.1f}")
            print(f"   Range: {aqi_stats['min']:.1f} - {aqi_stats['max']:.1f}")
            print(f"   Std Dev: {aqi_stats['std']:.1f}")
            
            if 'category_distribution' in aqi_stats:
                print(f"\n🏷️  AQI Categories:")
                for category, info in aqi_stats['category_distribution'].items():
                    print(f"   {category}: {info['percentage']}%")
        
        print(f"\n📈 Dataset Info:")
        print(f"   Records: {len(df):,}")
        print(f"   Features: {len(df.columns)}")
        print(f"   Date Range: {summary['overall']['date_range_days']} days" 
              if summary['overall']['date_range_days'] else "")
        
        print(f"\n📁 Complete analysis available in: {self.results_dir}")
        print("=" * 80)

# ==================== MAIN EXECUTION ====================
def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("🎯 INTELLIGENT EDA ANALYSIS FOR AQI KARACHI")
    print("=" * 80)
    
    try:
        # Initialize analyzer
        analyzer = EDAAnalyzer()
        
        # Run analysis
        analyzer.run()
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\n🔧 Debugging Information:")
        print(f"   Python: {sys.version}")
        print(f"   Working Directory: {os.getcwd()}")
        print(f"   Script Location: {__file__}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()