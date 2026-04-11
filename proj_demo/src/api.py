from fastapi import FastAPI, HTTPException
import pandas as pd
import pandera as pa

import mlflow
import joblib
from pathlib import Path

import numpy as np

from .schema import InputSchema, IrisRequest, IrisResponse

app = FastAPI(title='Iris Classification API')

best_model = None

model_metadata = None

@app.on_event("startup")
def load_best_model():
    global best_model, model_metadata

    models_dir = Path(__file__).parents[1] / "models"
    models_path = models_dir / "best_model.pkl"
    metadata_path = models_dir / "model_metadata.pkl"

    try:
        if models_path.exists():
            best_model = joblib.load(models_path)
            model_metadata = joblib.load(metadata_path)
            print(f"Succesfully loaded model: {model_metadata['model_name']}")
        else:
            print(f"Model not found at {models_path}")
    except Exception as e:
        print(f"Error loading best model: {e}")

@app.get("/")
def read_root():
    model_name = model_metadata['model_name'] if model_metadata else "None"
    return {"message" : f"Iris Classifier {model_name}"}

@app.get("/health")
def health_check():
    status = "healthy" if best_model is not None else "degraded"
    return {"status" : status}

@app.post("/predict", response_model=IrisResponse)
def predict(request: IrisRequest):
    data = {
            "sepal_length" : [request.sepal_length],
            "sepal_width" : [request.sepal_width],
            "petal_length" : [request.petal_length],
            "petal_width" : [request.petal_width],
    }
    df = pd.DataFrame(data)

    try:
        InputSchema.validate(df)
    except pa.errors.SchemaError as e:
        raise HTTPException(status_code=400, detail=f"Data is not valid: {e}")

    if best_model is None:
        raise HTTPException(status_code=503, details=f"model not found")

    try:
        prediction_idx = best_model.predict(df)[0]
        target_names = model_metadata['target_names']
        prediction = target_names[prediction_idx]
        prediction_proba = best_model.predict_proba(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    if probability >= 0.9:
        confidence = "High"
    elif probability >= 0.7:
        confidence = "Medium"
    else:
        confidence = "Low"

    try:
        if mlflow.active_run():
            mlflow.end_run()
        with mlflow.start_run(run_name='api_prediction'):
            mlflow.log_params(data)

            mlflow.log_metric("prediction_probability", probability)
            mlflow.set_tag("model_type", model_metadata['model_name'])
            mlflow.set_tag("prediction_class", prediction)
    except Exception as e:
        print(e)

    return {
        'prediction': prediction,
        'probability': probability,
        'confidence': confidence,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)



