# MLOps Zoomcamp/ 

**Week 2: Introduction to MLflow, ML experiments and model registry.**

# Introduction:

## Definitions:
+ ML experiment: the process of building an ML model; The whole process in which a Data Scientist creates and optimizes a model
+ Experiment run: each trial in an ML experiment; Each run is within an ML experiment
+ Run artifact: any file associated with an ML run: Examples include the model itself, package versions...etc; Each Artifact is tied to an Experiment
+ Experiment metadata: metadata tied to each experiment

## Experiment tracking:
Keeping track of all the relevant information from an ML experiment; varies from experiment to experiment.
Experiment tracking helps with *Reproducibility*, *Organization* and *Optimization*

Tracking experiments in spreadsheets helps but falls short in all the key points.

---

# MLflow:

![Mlflow](https://user-images.githubusercontent.com/24941662/171018109-5b3fe8a7-773f-4db0-a94e-ffd1caf703e5.png)


*"is an open source platform to manage the ML lifecycle, including experimentation, reproducibility, deployment, and a central model registry."*

It's a Python package with four main modules:
+ Tracking
+ Models
+ Model registry
+ Projects (Out of scope of the course)

## Tracking experiments with MLflow:

MLflow organizes experiments into runs and keeps track of any variables that may affect the model as well as its result; Such as: Parameters, Metrics, Metadata, the Model itself...

MLflow also automatically logs extra information  about each run such as: Source Code, Git Commit, Start and End time and Author.

## Installing MLflow:

pip: `pip install mlflow`

conda: `conda install -c conda-forge mlflow`

## Interacting with MLflow:

MLflow has different interfaces, each with their pros and cons. We introduce  the core functionalities of MLflow through the UI.

### MLflow UI:

To run the MLflow UI locally we use the command:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The backend storage is essential to access the features of MLflow, in this command we use a SQLite backend with the file `mlflow.db` in the current running repository. This URI is also given later to the MLflow Python API
`mlflow.set_tracking_uri`.

By accessing the provided local url we can access the UI. Within this UI we have access to MLflow features.

In addition to the backend URI, we can also add an artifact root directory where we store the artifacts for runs, this is done by adding a `--default-artifact-root` paramater:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns


### MLflow Tracking Client API:

In addition to the UI, an interface that is introduced in the course and used to automate processes is the Tracking API. Initialized through:
```python
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
```

the `client` is an object that allows managing experiments, runs, models and model registries (cf. Interacting with MLflow through the Tracking Client). See: https://www.mlflow.org/docs/latest/python_api/mlflow.tracking.html For more information on the interface.

## Creating new Experiments:

We create an experiment in the top left corner of the UI. (In this instance `nyc-taxi-experiment`).

Using the Python API we use `client.create_experiment("nyc-taxi-experiment")`.

## Tracking Single Experiment Runs with Mlflow in a Jupyter notebook or Python file:

In order to track experiment runs, we first initialize the mlflow experiment using the code:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")
```

where we set the tracking URI and the current experiment name. In case the experiment does not exist, it will be automatically created.

We can then track a run, we'll use this simple code snippet as a starting point:

```python
alpha = 0.01

lr = Lasso(alpha)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_val)

mean_squared_error(y_val, y_pred, squared=False)
```
We initialize the run using
```python
with mlflow.start_run():
```
and wrapping the whole run inside it.

We track the relevant information using  three mlflow commands:
+ `set_tag` for Metadata tags
+ `log_param` for logging model parameters
+ `log_metric` for logging model metrics

In this instance, we may set as Metadata tags the author name, the model parameters as the training and validation data paths and alpha, and set the metric as RMSE:

```python
with mlflow.start_run():
    mlflow.set_tag("developer","Qfl3x")
    
    mlflow.log_param("train-data-path", "data/green_tripdata_2021-01.parquet")
    mlflow.log_param("val-data-path", "data/green_tripdata_2021-02.parquet")
    
    alpha = 0.01
    mlflow.log_param("alpha", alpha)
    lr = Lasso(alpha)
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mlflow.log_metric("rmse", rmse)
```

In the MLflow UI, within the `nyc-taxi-experiment` we now have a run logged with our logged parameters, tag, and metric.

## Hyperparameter Optimizaiton Tracking:

By wrapping the `hyperopt` Optimization objective inside a `with mlflow.start_run()` block, we can track every optimization run that was ran by `hyperopt`. We then log the parameters passed by `hyperopt` as well as the metric as follows:

```python


import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

train = xgb.DMatrix(X_train, label=y_train)
valid = xgb.DMatrix(X_val, label=y_val)

def objective(params):
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

    return {'loss': rmse, 'status': STATUS_OK}

search_space = {
    'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
    'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
    'objective': 'reg:linear',
    'seed': 42
}

best_result = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=Trials()
)

```

In this block, we defined the search space and the objective than ran the optimizer. We wrap the training and validation block inside `with mlflow.start_run()` and log the used parameters using `log_params` and validation RMSE using `log_metric`.

In the UI we can see each run of the optimizer and compare their metrics and parameters. We can also see how different parameters affect the RMSE using Parallel Coordinates Plot, Scatter Plot (1 parameter at a time) and Contour Plot.

## Autologging: 

Instead of logging the parameters by "Hand" by specifiying the logged parameters and passing them. We may use the Autologging feature in MLflow. There are two ways to use Autologging; First by enabling it globally in the code/Notebook using 
```python
mlflow.autolog()
```

or by enabling the framework-specific autologger; ex with XGBoost:

```python
mlflow.xgboost.autolog()
```
Both must be done before running the experiments.

The autologger then not only stores the model parameters for ease of use, it also stores other files inside the `model` (can be specified) folder inside our experiment artifact folder, these files include:
+ `conda.yaml` and `requirements.txt`: Files which define the current envrionment for use with either `conda` or `pip` respectively
+ `MLmodel` an internal MLflow file for organization
+ Other framework-specific files such as the model itself


## Saving Models:

We may use MLflow to log whole models for storage (see Model Registry later), to do this we add a line to our `with mlflow.start_run()` block:

```python
mlflow.<framework>.log_model(model, artifact_path="models_mlflow")
```

where we replace the `<framework>` wih our model's framework (ex: `sklearn`, `xgboost`...etc).
The `artifact_path` defines where in the `artifact_uri` the model is stored.

We now have our model inside our `models_mlflow` directory in the experiment folder. (Using Autologging would store more data on parameters as well as the model. i.e: This is redundant when using the autologger)

## Saving Artifacts with the Model:

Sometimes we may want to save some artifacts with the model, for example in our case we may want to save the `DictVectorizer` object with the model for inference (subsequently testing as well). In that case we save the artifact as:
```python
mlflow.log_artifact("vectorizer.pkl", artifact_path="extra_artifacts")
```

Where `vectorizer.pkl` is the vectorizer stored in a Pickle file and `extra_artifacts` the folder within the artifacts of the model where the file is stored.

## Loading Models:

We can use the model to make predictions with multiple ways depending on what we need:
+ We may load the model as a Spark UDF (User Defined Function) for use with Spark Dataframes
+ We may load the model as a MLflow PyFuncModel structure, to then use to predict data in a Pandas DataFrame, NumPy Array or SciPy Sparse Array. The obtained interface is general for all models from all frameworks
+ We may load the model as is, i.e: load the XGBoost model as an XGBoost model and treat it as such

The first two methods are explained briefly in the MLflow artifacts page for each run, for the latter we may use (XGBoost example):
```python
logged_model = 'runs:/9245396b47c94513bbf9a119b100aa47/models' # Model UUID from the MLflow Artifact page for the run

xgboost_model = mlflow.xgboost.load_model(logged_model)
```
the resultant `xgboost_model` is an XGBoost `Booster` object which behaves like any XGBoost model. We can predict as normal and even use XGBoost Booster functions such as `get_fscore` for feature importance.


## Model Registry:

Just as MLflow helps us store, compare and deal with ML experiment runs. It also allows us to store Models and categoerize them. While it may be possible to store models in a folder structure manually, doing this is cumbersome and leaves us open to errors. MLflow deals with this using the Model Registry, where models may be stored and labeled depending on their status within the project.

### Storing Models in the Registry:

In order to register models using the UI, we select the run whose model we want to register and then select "Register Model". There we may either create a new model registry or register the model into an existing registry. We can view the registry and the models therein by selecting the "Models" tab in the top and selecting the registry we want.

### Promoting and Demoting Models in the registry:

Models in the registry are labeled either as Staging, Production or Archive. Promoting and demoting a model can be done by selecting the model in the registry and selecting the stage of the model in the drop down "Stage" Menu at the top.

## Interacting with MLflow through the Tracking Client:

In order to automate the process of registering/promoting/demoting models, we use the Tracking Client API initialized as described above:

```python
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
```

we can then use the client to interface with the MLflow backend as with the UI.

---
### Selecting runs:

We can search for runs by ascending order of metric score using the API by:

```python
from mlflow.entities import ViewType

runs = client.search_runs(
    experiment_ids='1',    # Experiment ID we want
    filter_string="metrics.rmse < 7",
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=5,
    order_by=["metrics.rmse ASC"]
)
```
We can then get information about the selected runs from the resulting `runs` enumerator:
```python
for run in runs:
    print(f"run id: {run.info.run_id}, rmse: {run.data.metrics['rmse']:.4f}")
```

### Interacting with the Model Registry:

We can add a run model to a registry using:
```python
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

run_id = "9245396b47c94513bbf9a119b100aa47"
model_uri = f"runs:/{run_id}/models"
mlflow.register_model(model_uri=model_uri, name="nyc-taxi-regressor")
```

we can get the models  in a model registry:
```python
model_name = "nyc-taxi-regressor"
latest_versions = client.get_latest_versions(name=model_name)

for version in latest_versions:
    print(f"version: {version.version}, stage: {version.current_stage}")
```

promote a model to staging:
```python
model_version = 4
new_stage = "Staging"
client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage=new_stage,
    archive_existing_versions=False
)
```

update the description of a model:
```python
from datetime import datetime

date = datetime.today().date()
client.update_model_version(
    name=model_name,
    version=model_version,
    description=f"The model version {model_version} was transitioned to {new_stage} on {date}"
)
```

these can then be used to automate the promotion of packages into production or the archival of older models.
