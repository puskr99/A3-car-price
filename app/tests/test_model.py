import os
import joblib
import pandas as pd
import mlflow
import mlflow.pyfunc


# model_path = "app/model/car_price_predictor"
scalar_model_path  = "app/model/scaler.pkl"

test = pd.DataFrame([[6.42, 10, 120000, 16, 1298, 88]],
                    columns=['name', 'year', 'km_driven', 'mileage', 'engine', 'max_power'])

scaler_model = joblib.load(scalar_model_path)


mlflow.set_tracking_uri("https://mlflow.ml.brain.cs.ait.ac.th/")
os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"


model_name = "st125098-a3-model"
client = mlflow.tracking.MlflowClient()
latest_version = client.get_latest_versions(model_name, stages=["Staging"])[0].version

# get latest model from staging version.
print("Latest version", latest_version)
model_mlflow = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{latest_version}")
print("Loaded model", model_mlflow)


def test_input():
    try:
        data = scaler_model.fit_transform(test)
        predicted_value = model_mlflow.predict(data)
        print("Predicted valuee is", predicted_value)
    except:
        assert(False, "Input shape is incorrect. Try again..")


def test_predicted_shape():
    data = scaler_model.fit_transform(test)

    predicted_value = model_mlflow.predict(data)

    #predicted values can only be 0-3.
    assert predicted_value[0] in [0, 1, 2, 3]
    
    assert predicted_value.shape == (1,)