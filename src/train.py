import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
import sys

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np

EVAL_THRESHOLD = 0.70

def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    # 1.5.1: Đọc dữ liệu
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # Bonus 5: Cảnh báo lệch dữ liệu
    label_counts = df_train["target"].value_counts(normalize=True)
    # Convert numpy types to native Python types for JSON serialization
    distribution_dict = {int(k): float(v) for k, v in label_counts.to_dict().items()}
    print("Label distribution in training data:")
    for label, prop in distribution_dict.items():
        print(f"Class {label}: {prop:.2%}")
        if prop < 0.10:
            print(f"WARNING: Class {label} represents less than 10% of the training data! ({prop:.2%})")

    # 1.5.2: Tách đặc trưng và nhãn
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # 1.5.3: Bắt đầu MLflow run
    with mlflow.start_run():
        # 1.5.4: Ghi nhận tham số (hỗ trợ Bonus 2)
        model_type = params.get("model_type", "random_forest")
        model_params = params.get(model_type, {})
        
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(model_params)

        # 1.5.5: Khởi tạo và huấn luyện
        if model_type == "random_forest":
            model = RandomForestClassifier(**model_params, random_state=42)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(**model_params, random_state=42)
        elif model_type == "logistic_regression":
            model = LogisticRegression(**model_params, random_state=42)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        model.fit(X_train, y_train)

        # 1.5.6: Tính accuracy, f1_score
        preds = model.predict(X_eval)
        acc = float(accuracy_score(y_eval, preds))
        f1 = float(f1_score(y_eval, preds, average="weighted"))

        # 1.5.7: Ghi nhận chỉ số
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # Log phân phối nhãn cho Bonus 5
        for label, prop in distribution_dict.items():
             mlflow.log_metric(f"train_prop_class_{label}", prop)

        # 1.5.8: Log artifact
        mlflow.sklearn.log_model(model, "model")

        # 1.5.9: In kết quả
        print(f"Model: {model_type}")
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # 1.5.10: Lưu metrics
        os.makedirs("outputs", exist_ok=True)
        metrics = {
            "accuracy": acc, 
            "f1_score": f1,
            "distribution": distribution_dict
        }
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics, f)

        # Bonus 3: Báo cáo hiệu suất tự động
        cm = confusion_matrix(y_eval, preds)
        cr = classification_report(y_eval, preds, target_names=["thấp", "trung_bình", "cao"], zero_division=0)
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(cr)
        
        with open("outputs/report.txt", "w", encoding="utf-8") as f:
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(cm) + "\n\n")
            f.write("Classification Report:\n")
            f.write(cr)

        # 1.5.11: Lưu model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc

if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
