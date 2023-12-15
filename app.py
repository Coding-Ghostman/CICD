from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

class CarPriceModel:
    def __init__(self, model=None, scaler=None):
        self.model = model
        self.scaler = scaler

    def load_model(self, model_path='model/car_price_prediction_model.pkl'):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

    def preprocess_input(self, input_data):
        input_df = pd.DataFrame(input_data, index=[0])
        input_scaled = self.scaler.transform(input_df)
        return input_scaled

    def predict_price(self, input_data):
        input_scaled = self.preprocess_input(input_data)
        return self.model.predict(input_scaled)[0]

car_model = CarPriceModel()
car_model.load_model()
with open('model/scaler.pkl', 'rb') as scaler_file:
    car_model.scaler = pickle.load(scaler_file)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None

    if request.method == 'POST':
        try:
            input_data = request.form.to_dict()
            age = float(input_data['age'])
            mileage = float(input_data['mileage'])
            prediction = car_model.predict_price({'Age': age, 'Mileage': mileage})
        except Exception as e:
            error = str(e)
    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':

    app.run(host="0.0.0.0", port=8080) #for deployment run
    # app.run(host="127.0.0.1", port=8080,debug=True) # for local run