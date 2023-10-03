import pandas as pd

from transformers import pipeline
from GoogleNews import GoogleNews
from flask import Flask, render_template, request

from functions import *

# Create a Flask web app instance
app = Flask(__name__)

# Initialize GoogleNews instance
googlenews = GoogleNews()
googlenews.enableException(True)

# Initialize text models
sentimentizer = pipeline("sentiment-analysis", model ='distilbert-base-uncased-finetuned-sst-2-english')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Define a route for the root URL ("/")
@app.route("/", methods=["GET", "POST"])
def index():
    user_input = ""
    
    if request.method == "POST":
        user_input = request.form["user_input"]

    headlines = get_google_news_headlines(googlenews, user_input)

    if headlines:
        top_n = "\nTop {} headlines pulled...".format(len(headlines))

        sentiments, avg_sentiment, maxSentiment, maxSentimentIdx, minSentiment, minSentimentIdx = report_sentiment(sentimentizer, headlines)

        high_text = '\nHighest rated sentiment of "{}" given to the following headline: \n"{}"'.format(maxSentiment, headlines[maxSentimentIdx])
        low_text = '\nLowest rated sentiment of "{}" given to the following headline: \n"{}"'.format(minSentiment, headlines[minSentimentIdx])

        df = pd.DataFrame({'Sentiments': sentiments, 'Headlines': headlines})
        df.sort_values(by='Sentiments', inplace=True)

        summary = summarize_headlines(summarizer, df['Headlines'])

    else:
        top_n = "No headlines found."
        high_text, low_text = "", ""
        summary = ""
    
    return render_template("index.html", user_input=user_input)

# Run the app if this script is executed
if __name__ == "__main__":
    app.run()