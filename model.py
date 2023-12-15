import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle

class CarPriceModel:
    def __init__(self, model=None, scaler=None):
        self.model = model
        self.scaler = scaler

    def train_model(self, X_train, y_train):
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = LinearRegression()
        self.model.fit(X_train_scaled, y_train)

    def evaluate_model(self, X_test, y_test):
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f'Mean Squared Error: {mse}')
        print(f'R-squared: {r2}')
        return r2

    def save_model(self, file_path='model/car_price_prediction_model.pkl'):
        with open(file_path, 'wb') as file:
            pickle.dump(self.model, file)
        with open('model/scaler.pkl', 'wb') as scaler_file:
            pickle.dump(self.scaler, scaler_file)
        print(f'Model saved as {file_path}')

def display_accuracy(r2):
    print(f'Model Accuracy (R-squared): {r2}')

def main():
    df = pd.read_csv('data/carprices.csv')
    X = df[['Age', 'Mileage']]
    y = df['Sell Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    car_model = CarPriceModel()
    car_model.train_model(X_train, y_train)
    r2 = car_model.evaluate_model(X_test, y_test)
    car_model.save_model()
    display_accuracy(r2)

if __name__ == '__main__':
    main()
