"""
Quick Prediction Runner - SIMPLE DIRECT IMPORT
"""
import os
import sys

# Get the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add it to Python path
sys.path.insert(0, script_dir)

print(f"📁 Script directory: {script_dir}")
print(f"📄 Files here: {os.listdir(script_dir)}")

# Direct import - should work since they're in the same directory
try:
    import prediction_service
    print("✅ Imported prediction_service module")
    
    # Create instance
    service = prediction_service.PredictionService()
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nCreating a minimal service instead...")
    
    # Create a minimal service as fallback
    class MinimalPredictionService:
        def generate_quick_predictions(self):
            print("⚠️ Using minimal prediction service")
            print("This is a fallback - check your prediction_service.py file")
            return False
    
    service = MinimalPredictionService()

def main():
    print("=" * 60)
    print("🎯 Quick Prediction Runner")
    print("=" * 60)
    
    print("\n🔄 Generating predictions...")
    success = service.generate_quick_predictions()
    
    if success:
        print("\n✅ Success!")
    else:
        print("\n❌ Failed")
    
    print("=" * 60)

if __name__ == "__main__":
    main()