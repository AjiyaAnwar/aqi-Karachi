"""
Quick Prediction Runner - For manual updates
"""
import sys
import os
from model_training.prediction_service import PredictionService

def main():
    print("=" * 60)
    print("🎯 Quick Prediction Update")
    print("=" * 60)
    
    service = PredictionService()
    
    # Check current status
    is_fresh, message, timestamp = service.check_prediction_freshness()
    
    print(f"Current predictions: {message}")
    if timestamp:
        print(f"Last updated: {timestamp}")
    
    if is_fresh:
        print("\n✅ Predictions are already fresh!")
        print("Run anyway? (y/n): ", end="")
        choice = input().strip().lower()
        if choice != 'y':
            print("Exiting...")
            return
    
    print("\n🔄 Generating fresh predictions...")
    success = service.generate_quick_predictions()
    
    if success:
        print("\n✅ Predictions updated successfully!")
    else:
        print("\n❌ Failed to update predictions")
    
    print("=" * 60)

if __name__ == "__main__":
    main()