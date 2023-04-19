import mlflow

mlflow.set_tracking_uri("http://mlflow:5000")

with mlflow.start_run(run_name='Wolf Experiment'):
    mlflow.log_param("t1", 10)
    mlflow.log_metric("m1", 10)
    mlflow.log_metric("m1", 20)