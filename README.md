# Stock-Price-Predictor-Web-v1

## Overview

`Stock-Price-Predictor-Web-v1` is a web application designed to forecast stock prices using a Recurrent Neural Network (RNN) model. The application is built with Flask for the backend and uses HTML, CSS, and JavaScript for the frontend. It allows users to input parameters and receive stock price predictions based on historical data. This project is built as an improvement of older project with improvement:
1. Using Yahoo API to pull data instead of using CSV
2. Allow user to choose stock company, time step and the duration of the stock that user want

## Features

- **Stock Price Prediction**: Forecast future stock prices using an RNN model.
- **Customizable Parameters**: Input `Time_Step` and forecast duration to generate predictions.
- **Interactive Interface**: User-friendly web interface developed with Flask, HTML, CSS, and JavaScript.

## Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Model**: Recurrent Neural Network (RNN)
- **AI libraries**: Numpy, Pandas, Scikit-learn, Pytorch, Keras
- **API**: Yahoo! Finance
## Special Thanks
Special thanks for Mr Sumit Aggarwal as my interviewer at MorningStar last year. When I was interviewed at MorningStar before, he asked me how to improve my old RNN-stock-predictor project by download multiple datas at the large scale. I could not answer that question and he gave me his solution to use YahooAPI. 

## Changelog

### [1.0.0] - 2024-09-11
- Initial release of Stock-Price-Predictor-Web-v1
- Implemented basic functionality for stock price prediction using RNN
- Created a simple web interface for user input and display of results

---
