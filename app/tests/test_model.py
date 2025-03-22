import os
import joblib
import pandas as pd

test = pd.DataFrame([[6.42, 10, 120000, 16, 1298, 88]],
                    columns=['name', 'year', 'km_driven', 'mileage', 'engine', 'max_power'])
model_path = "app/model/A1-car_price_predictor"


def test_input():
    assert os.path.exists(model_path), f"Model file not found: {model_path}"

    model = joblib.load(model_path)
    try:
        predicted_value = model.predict(test)
    except:
        assert(False, "Shape is incorrect.")


def test_predicted_shape():
    assert os.path.exists(model_path), f"Model file not found: {model_path}"

    model = joblib.load(model_path)
    
    predicted_value = model.predict(test)

    assert predicted_value[0] > 0
    assert predicted_value.shape == (1,)