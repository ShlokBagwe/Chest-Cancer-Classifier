from ChestCancer import logger
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import mlflow
from sklearn.linear_model import ElasticNet
import mlflow.sklearn
import logging
import numpy as np
import pandas as pd

import os
import sys


dagshub.init(repo_owner='shlokbagwe.tech', repo_name='Chest-Cancer-Classifier', mlflow=True)

def evaluate(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    return r2, mae, rmse

if __name__ == "__main__":
    np.random.seed(42)

    csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"

    try:
        df = pd.read_csv(csv_url, sep = ";")
    except Exception as e:
        logger.exception("Unable to load the data")

    
    X = df.drop('quality', axis=1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    
    with mlflow.start_run():
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2, mae, rmse = evaluate(y_test, y_pred)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        remote_server_uri = "https://dagshub.com/shlokbagwe.tech/Chest-Cancer-Classifier.mlflow"
        mlflow.set_tracking_uri(remote_server_uri)

        mlflow.sklearn.log_model(
            model,
            "model", 
            registered_model_name="Elasticnet_Model"
        )

        