import os
import json
import numpy as np
import pandas as pd
from src.train import train

FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]

def _make_temp_data(tmp_path):
    """
    Creates a small dataset with the same Wine Quality schema for testing.
    pytest provides `tmp_path` as a temporary directory that is automatically cleaned up.
    """
    rng = np.random.default_rng(42)
    n = 200
    
    # Create feature matrix X of shape (n, 12) with random values in [0, 1)
    X = rng.random((n, len(FEATURE_NAMES)))
    
    # Create target array y of shape (n,) with random integers in [0, 3)
    y = rng.integers(0, 3, size=n)
    
    # Create DataFrame from X and y
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y
    
    # Save first 160 rows as train.csv and last 40 rows as eval.csv
    train_path = os.path.join(tmp_path, "train.csv")
    eval_path = os.path.join(tmp_path, "eval.csv")
    
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)
    
    return str(train_path), str(eval_path)

def test_train_returns_float(tmp_path):
    """Verifies that train() returns a float in the range [0, 1]."""
    train_path, eval_path = _make_temp_data(tmp_path)
    params = {
        "model_type": "random_forest",
        "random_forest": {
            "n_estimators": 5,
            "max_depth": 2
        }
    }
    acc = train(params, data_path=train_path, eval_path=eval_path)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0

def test_metrics_file_created(tmp_path):
    """Verifies that outputs/metrics.json is created and contains accuracy and f1_score."""
    train_path, eval_path = _make_temp_data(tmp_path)
    params = {
        "model_type": "random_forest",
        "random_forest": {
            "n_estimators": 5,
            "max_depth": 2
        }
    }
    train(params, data_path=train_path, eval_path=eval_path)
    
    metrics_file = "outputs/metrics.json"
    assert os.path.exists(metrics_file)
    
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
        
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert "distribution" in metrics
    assert isinstance(metrics["accuracy"], float)
    assert isinstance(metrics["f1_score"], float)

def test_model_file_created(tmp_path):
    """Verifies that models/model.pkl is created."""
    train_path, eval_path = _make_temp_data(tmp_path)
    params = {
        "model_type": "random_forest",
        "random_forest": {
            "n_estimators": 5,
            "max_depth": 2
        }
    }
    train(params, data_path=train_path, eval_path=eval_path)
    
    model_file = "models/model.pkl"
    assert os.path.exists(model_file)
