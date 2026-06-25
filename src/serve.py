from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

app = FastAPI()

# Read bucket from environment variable
AWS_BUCKET = os.environ.get("CLOUD_BUCKET")
S3_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")

def download_model():
    """Downloads model.pkl from S3 to local path."""
    if not AWS_BUCKET:
        raise ValueError("CLOUD_BUCKET environment variable is not set!")
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Downloading model from S3 bucket '{AWS_BUCKET}' key '{S3_MODEL_KEY}'...")
    s3 = boto3.client("s3")
    s3.download_file(AWS_BUCKET, S3_MODEL_KEY, MODEL_PATH)
    print("Model downloaded successfully.")

# Call download_model when the server starts
download_model()
model = joblib.load(MODEL_PATH)

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    """Health check endpoint to verify server is running."""
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Prediction endpoint.
    Expects 12 features representing chemical properties of wine.
    Returns predicted quality class (0, 1, or 2) and its textual label.
    """
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")
    
    try:
        prediction = int(model.predict([req.features])[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")
        
    labels = {0: "thấp", 1: "trung_bình", 2: "cao"}
    return {
        "prediction": prediction,
        "label": labels.get(prediction, "unknown")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
