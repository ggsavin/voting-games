import mlflow
import time

mlflow_uri = "http://mlflow:5000"

mlflow.set_tracking_uri(mlflow_uri)

experiment = mlflow.get_experiment_by_name("Test Experiment 1")

if experiment == None:
    experiment_id = mlflow.create_experiment(
        "Test Experiment 1",
        tags={"version": "v1", "priority": "P1"},
    )
else:
    experiment_id = experiment.experiment_id

with mlflow.start_run(
    run_name="test run",
    experiment_id=experiment_id,
    tags={"version": "v1", "priority": "P1"},
    description="The description of the test run"):

    mlflow.log_metric("metric1", 20)
    mlflow.log_metric("metric1", 20)