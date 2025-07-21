# models/verify_models.py
import joblib
import os

# Get models directory
current_dir = os.path.dirname(os.path.abspath(__file__))
model_files = os.listdir(current_dir)

print("Trained models:")
for file in model_files:
    if file.endswith('.pkl'):
        print(f"- {file}")
        try:
            model = joblib.load(os.path.join(current_dir, file))
            print(f"  Model type: {type(model).__name__}")
            if hasattr(model, 'n_features_in_'):
                print(f"  Features: {model.n_features_in_}")
        except Exception as e:
            print(f"  Error loading: {str(e)}")