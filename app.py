from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import re
import os
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle


IMAGE_FOLDER = os.path.join('static', 'img_pool')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def init():
    global model,graph
    graph = tf.compat.v1.get_default_graph()


@app.route('/', methods=['GET', 'POST'])
def home():

    return render_template("index.html")

@app.route('/sentiment_prediction', methods = ['POST', "GET"])
def sent_anly_prediction():
    if request.method=='POST':
        text = request.form['text']

        
        tw = tokenizer.texts_to_sequences([text])
        tw = sequence.pad_sequences(tw,maxlen=200)

        
        # word_to_id = imdb.get_word_index()
        # strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        # text = text.lower().replace("<br />", " ")
        # text=re.sub(strip_special_chars, "", text.lower())

        # words = text.split() #split string into a list
        # x_test = [[word_to_id[word] if (word in word_to_id and word_to_id[word]<=20000) else 0 for word in words]]
        # x_test = sequence.pad_sequences(x_test, maxlen=200) # Should be same which you used for training data
        # vector = np.array([x_test.flatten()])

        
        with graph.as_default():
            # load the pre-trained Keras model
            model = load_model('sentiment_analysis.h5')

            # probability = model.predict(np.array([vector][0]))[0][0]
            # prediction = (model.predict(np.array([vector][0])) > 0.5).astype("int32")

            probability = model.predict(tw)[0][0]
            prediction = int(model.predict(tw).round().item())
            
        if prediction == 0:
            sentiment = 'Negative'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'sad.gif')
        else:
            sentiment = 'Positive'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'happy.gif')
    return render_template('index.html', text=text, sentiment=sentiment, probability=probability, image=img_filename)


if __name__ == "__main__":
    init()
    app.run(debug=True)
