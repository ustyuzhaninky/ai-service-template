import json
import os

from flask import Flask, request

from train import prepare_training_dataset
from train import generate_text
from train import get_model

os.environ["MODEL_DIR"] = ''
os.environ["MODEL_FILE"] = 'model.tf'

MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

print("Loading model from: {}".format(MODEL_PATH))
model = get_model(MODEL_PATH)

app = Flask(__name__)

@app.route('/line/<int:Line>')
def line(Line):
    with open('./test.txt', 'rt') as file:
        file_data = file.read()
    return json.dumps(file_data[Line])

@app.route('/prediction/', methods=['POST', 'GET'])
def prediction():

    query_string = [str(request.args.get('query'))]
    try: 
        n_characters = int(request.args.get('lenght'))
    except:
        n_characters = 100
    prediction = generate_text(
        model,
        n_characters=n_characters,
        query=query_string)

    return prediction

@app.route('/score', methods=['POST', 'GET'])
def score():

    with open('./test.txt', 'rt') as file:
        data_test = file.read()
    data = prepare_training_dataset(data_test)
    result = model.evaluate(data)
    
    return dict(zip(model.metrics_names, result))

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
