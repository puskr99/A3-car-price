from flask import Flask, render_template, request
import numpy as np
import cloudpickle
import joblib

app = Flask(__name__)

# Car brands for dropdown (not part of X, just for UI)
car_brands = {
    'Ashok': 6,
    'Ambassador': 3,
    'Audi': 26,
    'BMW': 30,
    'Chevrolet': 4,
    'Datsun': 7,
    'Daewoo': 2,
    'Fiat': 5,
    'Force': 19,
    'Isuzu': 23,
    'Jaguar': 27,
    'Jeep': 24,
    'Kia': 21,
    'Land': 29,
    'Lexus': 31,
    'Opel': 1,
    'Tata': 8,
    'Maruti': 9,
    'Hyundai': 10,
    'Peugeot': 0,
    'Renault': 11,
    'Ford': 14,
    'Honda': 15,
    'Skoda': 16,
    'Mahindra': 17,
    'Mercedes-Benz': 25,
    'MG': 22,
    'Mitsubishi': 18,
    'Nissan': 12,
    'Toyota': 20,
    'Volvo': 28,
    'Volkswagen': 13,
}

# Fuel options for dropdown
fuel_options = ["Petrol", "Diesel"]

# Ranges for form validation
bought_year_range = [1980, 2024]
max_power_range = [1, 99999]
engine_cc_range = [1, 99999]
mileage_range = [1, 999]

# Load models and scaler
with open("../app/model/car_price_predictor", "rb") as f:
    predictor_a2 = cloudpickle.load(f)

scaler_fit_model = joblib.load("./model/scaler.pkl")
predictor_a1 = joblib.load("./model/A1-car_price_predictor")

@app.route("/")
def index():
    return render_template(
        'index.html',
        car_brands=car_brands,
        fuel=fuel_options,
        bought_year_range=bought_year_range,
        max_power_range=max_power_range,
        mileage_range=mileage_range,
        engine_cc_range=engine_cc_range,
        selected_data={}
    )

# @app.route('/a3', methods=['GET', 'POST'])
# def car_price_prediction_a2():
#     form_datas = {}
#     pred_selling_price = None

#     if request.method == 'POST':
#         # Collect form data
#         car_brand = request.form.get("car_brand")
#         fuel_input = request.form.get("fuel")  # "Petrol" or "Diesel"

#         # Map fuel to one-hot encoded columns
#         petrol = 1 if fuel_input == "Petrol" else 0
#         diesel = 1 if fuel_input == "Diesel" else 0

#         Low= 1 if car_brand == "Low" else 0
#         Mid= 1 if car_brand == "Mid" else 0
#         High= 1 if car_brand == "High" else 0

#         form_datas = {
#             "year": 2025 - int(request.form.get("bought_year")),
#             "km_driven": float(request.form.get("km_driven")),
#             "mileage": float(request.form.get("mileage")),
#             "engine": float(request.form.get("engine")),
#             "max_power": float(request.form.get("max_power")),
#             "Low": Low,
#             "Mid": Mid,
#             "High": High,
#             "Petrol": petrol,
#             "Diesel": diesel
#         }

#         # Predict selling price and bin it
#         pred_selling_price = get_predicted_selling_price_a2(form_datas)

#     return render_template(
#         'index.html',
#         car_brands=car_brands,
#         fuel=fuel_options,
#         bought_year_range=bought_year_range,
#         max_power_range=max_power_range,
#         mileage_range=mileage_range,
#         engine_cc_range=engine_cc_range,
#         selling_price=pred_selling_price,
#         selected_data=form_datas
#     )

# def get_predicted_selling_price_a2(form_data):
#     input_data = [
#         form_data["year"],
#         form_data["km_driven"],
#         form_data["mileage"],
#         form_data["engine"],
#         form_data["max_power"],
#         form_data["Low"],
#         form_data["Mid"],
#         form_data["High"],
#         form_data["Petrol"],
#         form_data["Diesel"]
#     ]

#     reshaped_array = np.array(input_data).reshape(1, -1)
#     final_data = scaler_fit_model.transform(reshaped_array)

#     try:
#         selling_price = predictor_a2.predict(final_data)
#         print("Selling price", selling_price)
#         return str(selling_price[0])
#     except Exception as e:
#         print(f"Prediction error: {e}")
#         return -1


@app.route('/a1', methods=['GET', 'POST'])
def car_price_prediction_a1():

    form_datas = {}
    pred_selling_price = None

    if request.method == 'POST':
        form_datas = {
            "car_brand": car_brands[request.form.get("car_brand")],
            "bought_year": request.form.get("bought_year"),
            "km_driven": request.form.get("km_driven"),
            # "fuel": fuel_encoded[fuel.index(request.form.get("fuel"))],
            # "seller": seller_type_encoded[seller_type.index(request.form.get("seller"))],
            # "transmission": transmission_encoded[transmission.index(request.form.get("transmission"))],
            # "owner": owner_mapping[request.form.get("owner")],
            "mileage": request.form.get("mileage"),
            "engine": request.form.get("engine"),
            "max_power": request.form.get("max_power"),
            # "seats": request.form.get("seats"),
        }

        pred_selling_price = get_predicted_selling_price_a1(form_datas.values())
    
    return render_template(
        'index.html',
        car_brands = car_brands,
        # owner = owner,
        # fuel = fuel,
        bought_year_range = bought_year_range,
        # transmission = transmission,
        max_power_range = max_power_range,
        # seller_type = seller_type,
        mileage_range = mileage_range,
        # seats_range = seats_range,
        engine_cc_range = engine_cc_range,
        selling_price = pred_selling_price,
        selected_data = form_datas
    )


def get_predicted_selling_price_a1(p_user_data):
    selling_price = -1

    # scalar = StandardScaler()
    reshaped_array = np.array(list(p_user_data)).reshape(1, -1)
    final_data = scaler_fit_model.transform(reshaped_array)

    try:
        selling_price = predictor_a2.predict(final_data)
        return selling_price[0]
    except:
        selling_price = -1

    return selling_price


if __name__ == '__main__':
    app.run(debug=True)