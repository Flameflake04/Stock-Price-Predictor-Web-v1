from flask import Flask, request, render_template
from stock_model_RNN import RNNModel

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_name = request.form.get('stock_name')
        stock_period = request.form.get('stock_period')
        stock_time_step = int(request.form.get('stock_time_step'))

        # Create and train the model
        model = RNNModel(stock_name, stock_period, stock_time_step)
        model.visualize()

        # Pass results to the result page
        return app.send_static_file('result.html')
    return app.send_static_file("index.html")

if __name__ == "__main__":
    app.run(debug=True)