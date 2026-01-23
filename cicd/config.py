# cicd/config.py
import os

# Project structure
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path configurations
PATHS = {
    'data_collection': os.path.join(PROJECT_ROOT, 'data_pipeline', 'collect_historical.py'),
    'feature_engineering': os.path.join(PROJECT_ROOT, 'data_pipeline', 'features.py'),
    'model_training': os.path.join(PROJECT_ROOT, 'model_training', 'runallmodels.py'),
    'time_series_training': os.path.join(PROJECT_ROOT, 'model_training', 'train_time_series_models.py'),
    'eda': os.path.join(PROJECT_ROOT, 'notebook', 'eda.py'),
    'tests': os.path.join(PROJECT_ROOT, 'testdata.py'),
    'dashboard': os.path.join(PROJECT_ROOT, 'dashboard', 'dashboard.py'),
}

# Schedule configurations
SCHEDULE = {
    'data_collection': '00:00',
    'feature_engineering': '01:00',
    'eda': '02:00',
    'model_training_hours': 3,
    'dashboard_hours': 4,
    'test_hours': 12,
    'full_pipeline_day': 'sunday',
    'full_pipeline_time': '03:00',
}