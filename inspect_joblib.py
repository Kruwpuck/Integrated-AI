import joblib
import sys

try:
    path = 'model_lgbm_machinelearning.joblib'
    pipeline = joblib.load(path)
    print(f"Type: {type(pipeline)}")
    if isinstance(pipeline, dict):
        print(f"Keys: {pipeline.keys()}")
        for k, v in pipeline.items():
            print(f"Key: {k}, Type: {type(v)}")
    else:
        print(f"Content: {pipeline}")
        if hasattr(pipeline, 'steps'):
            print("Steps:")
            for step in pipeline.steps:
                print(step)
except Exception as e:
    print(f"Error: {e}")
