import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import re, emoji, string
import pyarabic.araby as ar
from sklearn.feature_extraction.text import TfidfTransformer , TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("C:\\Users\\SaWa\\Untitled Folder\\model.pkl", "rb"))
pipe = pickle.load(open("C:\\Users\\SaWa\\Untitled Folder\\pipe.pkl", "rb"))
@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    def data_cleaning(text):
        text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)  # Removing all links
        text = re.sub(r'^http?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"https\S+", "", text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub("(\s\d+)", "", text)
        text = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", "", text)
        text = re.sub("\d+", " ", text)
        text = ar.strip_tashkeel(text)
        text = ar.strip_tatweel(text)
        text = text.replace("#", " ");
        text = text.replace("@", " ");
        text = text.replace("_", " ");
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        text = emoji.replace_emoji(text, replace="")  # removing emojis
        text = text.replace("آ", "ا")
        text = text.replace("إ", "ا")
        text = text.replace("أ", "ا")
        text = text.replace("ؤ", "و")
        text = text.replace("ئ", "ي")
        text = re.sub(r'[a-zA-Z]', '', text)  # removing all english chars
        text = re.sub(r'[^\w\s]', '', text)  # removing all punctuation
        return text


    text1 =  str(request.form['Texts'])
    text = data_cleaning(text1)
    trans_text = pipe.transform([text])
    prediction = model.predict(trans_text)
    return render_template("index.html", prediction_text = "The Dialect detected is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)