from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import uuid
import time
import os
import shutil
from matplotlib.figure import Figure


# Loading models
model = pickle.load(open('lr_tfidf_model.pkl', 'rb'))
tfidf_vect = pickle.load(open('tfidf.pkl', 'rb'))
custom_model = pickle.load(open('custom_model.pkl', 'rb'))
ps = PorterStemmer()
analyzer = SentimentIntensityAnalyzer()


def delete_image(folder_path):
   for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
# Routes
app = Flask(__name__)
image_folder = 'static'
delete_image(image_folder)


def find_urls(text):
  url_pattern= re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
  urls = re.findall(url_pattern,text)
  if len(urls)==0:
    return 0
  else:
    return urls
  
def find_pronoun(text):
  pronoun_pattern = re.compile(r'\b(I|me|My|my|you|You|he|He|him|she|She|her|it|It|we|We|us|they|They|them)\b')
  pronouns = re.findall(pronoun_pattern,text)
  if len(pronouns)==0:
    return 0
  else:
    return len(pronouns)
  
def spec_count(text):
  spec=text.count('@!?') if text.count('@!?')>0 else 0
  return spec

def get_compound_score(text):
    scores = analyzer.polarity_scores(text)
    compound_score = scores['compound']
    return compound_score

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

def predict(text):
    # Same prediction logic as before
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    review_vect = tfidf_vect.transform([review])
    feature_names = tfidf_vect.get_feature_names_out()
    prediction = 'FAKE' if model.predict(review_vect) == 0 else 'REAL'
    return prediction, feature_names

def predict_custom(text):
    data = {'url_count': [find_urls(text)], 'pronoun_counts': [find_pronoun(text)],'sentiment_score': [get_compound_score(text)],'spe_char_count': [spec_count(text)]}
    data_df = pd.DataFrame(data)
    feature_names = data_df.columns
    return feature_names

def display_plot(test_model,feature_names):
    coefficients = test_model.coef_[0]  # Extract coefficients from the model
    # Sort features by importance
    sorted_idx = np.argsort(np.abs(coefficients))[::-1]
    # Plot the coefficients
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)

    # Plot data
    axis.barh(np.array(feature_names)[sorted_idx][:20], coefficients[sorted_idx][:20], color='#008080')

    # Labels and title
    axis.set_xlabel("Coefficient Value (Feature Importance)")
    axis.set_ylabel("Feature")

    # Invert y-axis for better readability
    axis.invert_yaxis()

    # Generate a unique filename using UUID
    unique_filename = f"static/cust-plot-{uuid.uuid4().hex}.png"

    # Save the figure with a unique filename
    fig.savefig(unique_filename, format="png", dpi=200, bbox_inches="tight")

    return unique_filename


def cust_display_plot(test_model, feature_names):
    coefficients = np.abs(test_model.coef_[0])  # Use absolute values for meaningful proportions
    sorted_idx = np.argsort(coefficients)[::-1]
    
    top_features = np.array(feature_names)[sorted_idx][:20]
    top_importance = coefficients[sorted_idx][:20]

    explode = (0.1, 0.2, 0.1,0.2)

    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    
    wedges, texts, autotexts = axis.pie(top_importance, labels=top_features, autopct='%1.1f%%',explode=explode, startangle=900, colors=['skyblue', 'lightcoral', 'gold', 'lightgreen'],pctdistance=0.50)
    axis.legend(wedges, top_features, title="Features", loc="best", bbox_to_anchor=(1, 0.5))

    unique_filename = f"static/tfidf-plot-{uuid.uuid4().hex}.png"
    fig.savefig(unique_filename, format="png", dpi=200, bbox_inches="tight")
    
    return unique_filename


@app.route('/', methods=['POST'])
def detection():
    data = request.form['text']
    prediction, feature_names = predict(data)
    cust_feature_names = predict_custom(data)

    image_url1 = display_plot(model, feature_names)
    image_url2=cust_display_plot(custom_model,cust_feature_names)

    # Return a JSON response with the result and image URL
    response = {
        'prediction': prediction,
        'image_url': image_url1,
        'cust_image_url': image_url2,
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
