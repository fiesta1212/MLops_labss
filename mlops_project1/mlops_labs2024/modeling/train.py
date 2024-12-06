import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import mlflow
from mlops_labs2024.create_bucket import create_bucket
from mlops_labs2024.data_upload import upload_file
from joblib import dump
import os
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt


argparser = argparse.ArgumentParser()
argparser.add_argument("-p", "--params", required=True, help="File path to params")
argparser.add_argument("-d", "--data_path", required=True, help="File path to data")

mlflow.set_tracking_uri("http://localhost:5000")
try:
    mlflow.create_experiment("Experiment", artifact_location="s3://mlflow")
except mlflow.MlflowException as e:
    print(e)
mlflow.set_experiment("Experiment")


def load_data(file_path: str):
    if not file_path.endswith(".csv"):
        raise ValueError(f"Wrong file type: {file_path}. Expected .csv")
    try:
        df = pd.read_csv(file_path, delimiter=",")
        X, y = df.drop(columns=["Churn"]), df["Churn"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise
    except Exception as e:
        print(f"Error while reading file: {e}")
        raise


def train_model_random_forest(X_train, X_test, y_train, y_test, params):
    with mlflow.start_run(run_name="Random Forest Classifier"):
        tuned_parameters = {}

        # Итерируемся по параметрам random_search
        for param in params:
            if "n_estimators" in param:
                _n_estimators_range = param["n_estimators"]["range"]
                tuned_parameters["n_estimators"] = randint(
                    _n_estimators_range[0], _n_estimators_range[1]
                )
            elif "max_depth" in param:
                _max_depth_range = param["max_depth"]["range"]
                tuned_parameters["max_depth"] = randint(
                    _max_depth_range[0], _max_depth_range[1]
                )
            elif "min_samples_split" in param:
                _min_samples_split_range = param["min_samples_split"]["range"]
                tuned_parameters["min_samples_split"] = randint(
                    _min_samples_split_range[0], _min_samples_split_range[1]
                )
            elif "min_samples_leaf" in param:
                _min_samples_leaf_range = param["min_samples_leaf"]["range"]
                tuned_parameters["min_samples_leaf"] = randint(
                    _min_samples_leaf_range[0], _min_samples_leaf_range[1]
                )

        rf_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            verbose=3,
            param_distributions=tuned_parameters,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=27),
            n_iter=20,
        )
        rf_search.fit(X_train, y_train)
        rf_best = rf_search.best_params_

        mlflow.log_param("n_estimators", rf_best.get("n_estimators"))
        mlflow.log_param("max_depth", rf_best.get("max_depth"))

        metrics = [f1_score, accuracy_score, precision_score, recall_score]
        metric_names = ["f1_score", "accuracy_score", "precision_score", "recall_score"]

        for name, metric in zip(metric_names, metrics):
            _metric_train = calculate_metric(
                rf_search.best_estimator_, X_train, y_train, metric
            )
            _metric_val = calculate_metric(
                rf_search.best_estimator_, X_test, y_test, metric
            )
            print(name + f" на тренировочной выборке: {_metric_train: .4f}")
            print(name + f" на валидационной выборке: {_metric_val: .4f}")
            mlflow.log_metric(name, _metric_train)
            mlflow.log_metric(name, _metric_val)

        y_pred = rf_search.best_estimator_.predict(X_test)
        report = classification_report(
            y_test, y_pred, target_names=np.array(["no", "yes"]), output_dict=True
        )
        report_path = "reports/classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f)

        create_bucket("mlflow")
        upload_file(
            "mlflow",
            f"artifacts/{mlflow.active_run().info.run_id}/classification_report.json",
            report_path,
        )

        fprate, tprate, _ = roc_curve(
            y_test, rf_search.best_estimator_.predict_proba(X_test)[:, 1]
        )
        plt.figure()
        plt.plot(fprate, tprate, color="g")
        plt.plot([0, 1], [0, 1], color="r", linestyle="--")
        plt.title("ROC curve")
        plt.ylabel("true positive rate, TPR")
        plt.xlabel("false positive rate, FPR")
        plt.grid(color="w")
        figure_path = "reports/figures/roc_curve.png"
        plt.savefig(figure_path)

        upload_file(
            "mlflow",
            f"artifacts/{mlflow.active_run().info.run_id}/roc_curve.png",
            figure_path,
        )

        model_name = (
            f"RandomForest_"
            f'{rf_best["n_estimators"]}_'
            f'{rf_best["max_depth"]}'
            f".joblib"
        )

        dump(rf_search.best_estimator_, os.path.join("models", model_name))
        create_bucket("model")
        upload_file(
            "model",
            f"experiments/{mlflow.active_run().info.run_id}/{model_name}",
            f"models/{model_name}",
        )


def calculate_metric(model_pipe, X, y, metric=f1_score):
    y_model = model_pipe.predict(X)
    return metric(y, y_model)


def prepare_params(**kwargs):
    params = {
        k0: {k1: kwargs.get(k0, {}).get(k1, v1) for k1, v1 in v0.items()}
        if type(v0) in {dict, DictConfig}
        else kwargs.get(k0, v0)
        for k0, v0 in kwargs.items()
    }
    return params


if __name__ == "__main__":
    args = argparser.parse_args()

    X_train, X_val, y_train, y_val = load_data(args.data_path)
    params = prepare_params(**OmegaConf.load(args.params))

    mlflow.set_experiment("Experiment")

    train_model_random_forest(X_train, X_val, y_train, y_val, params["random_search"])
