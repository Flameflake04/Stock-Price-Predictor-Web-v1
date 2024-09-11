import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


class RNNModel:
    def __init__(self, stock_name, stock_period, stock_time_step):
        self.stock_name = stock_name
        self.stock_period = stock_period
        self.stock_time_step = stock_time_step
        self.scaler = MinMaxScaler(feature_range=(0, 1))  

    def gather_data(self):
        stock = yf.Ticker(self.stock_name)
        stock_data = stock.history(period=self.stock_period)["Close"]
        stock_data_scaled = self.scaler.fit_transform(stock_data.values.reshape(-1, 1))
        return stock_data_scaled
    
    def create_sequences(self, data, time_steps):
        sequences = []
        labels = []
        for i in range(0, len(data) - time_steps):
            sequences.append(data[i:i+time_steps])
            labels.append(data[i+time_steps])
        return np.array(sequences), np.array(labels)

    def scale_data(self):
        self.data_cleaned = self.gather_data()
        X, y = self.create_sequences(self.data_cleaned, self.stock_time_step)
        train_size = int(X.shape[0] * 0.8)
        self.X_train, self.X_test = X[:train_size], X[train_size:]
        self.y_train, self.y_test = y[:train_size], y[train_size:]

    def set_up_model(self):
        self.scale_data() 
        self.model = Sequential()
        self.model.add(SimpleRNN(units=50, return_sequences=False, input_shape=(self.stock_time_step, 1)))
        self.model.add(Dense(units=1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(self.X_train, self.y_train, epochs=100, batch_size=32, validation_data=(self.X_test, self.y_test))
        self.y_pred = self.model.predict(self.X_test)

    def visualize(self):
        self.set_up_model()
        y_pred_scaled = self.scaler.inverse_transform(self.y_pred)
        y_test_scaled = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        r2 = r2_score(y_test_scaled, y_pred_scaled)
        
        print(f"R² Score: {r2}")
        plt.plot(y_test_scaled, label="Actual Stock Price")
        plt.plot(y_pred_scaled, label="Predicted Stock Price")
        plt.title("Stock Price Predictor Result")
        plt.xlabel("Time (day)")
        plt.ylabel("Close Index Price")

        plt.text(0.05, 0.95, f'R² Score: {r2:.2f}', transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        plt.legend()
        plt.show()
        


if __name__ == "__main__":
    model = RNNModel("AAPL", "5y", 10)
    model.visualize()
