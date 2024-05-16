import matplotlib
matplotlib.use('agg')
from flask import Flask, render_template, request
from model_module import predict_and_plot

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index', methods=['GET', 'POST'])
def index_page():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])  # Allow both GET and POST requests
def result():
    if request.method == 'POST':
        input_text = request.form['input_text']
        prediction, graph_url = predict_and_plot(input_text)
        return render_template('result.html', prediction=prediction, graph_url=graph_url)

    else:
        # Handle GET request (optional)
        return 'This route only accepts POST requests.'

if __name__ == '__main__':
    app.run(debug=True)
