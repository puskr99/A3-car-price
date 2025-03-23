import os
import joblib
import pandas as pd

model_path = "app/model/car_price_predictor"
scalar_model_path  = "app/model/scaler.pkl"

test = pd.DataFrame([[6.42, 10, 120000, 16, 1298, 88]],
                    columns=['name', 'year', 'km_driven', 'mileage', 'engine', 'max_power'])

scaler_model = joblib.load(scalar_model_path)


def test_input():
    assert os.path.exists(model_path), f"Model file not found: {model_path}"

    model = joblib.load(model_path)
    try:
        data = scaler_model.fit_transform(test)
        predicted_value = model.predict(data)
    except:
        assert(False, "Shape is incorrect.")


def test_predicted_shape():
    assert os.path.exists(model_path), f"Model file not found: {model_path}"

    model = joblib.load(model_path)

    data = scaler_model.fit_transform(test)

    predicted_value = model.predict(data)

    #predicted values can only be 0-3.
    assert predicted_value[0] in [0, 1, 2, 3]
    
    assert predicted_value.shape == (1,)