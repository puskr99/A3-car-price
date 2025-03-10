import os
import joblib
import pandas as pd


def test_model():
    model_path = "app/model/A1-car_price_predictor"

    assert os.path.exists(model_path), f"Model file not found: {model_path}"

    model = joblib.load(model_path)

    # Test data
    test = pd.DataFrame([[15, 2007, 120000, 16, 1298, 88]],
                         columns=['name', 'year', 'km_driven', 'mileage', 'engine', 'max_power'])
    
    predicted_value = model.predict(test)

    assert predicted_value[0] > 0
    assert predicted_value.shape == (1,)