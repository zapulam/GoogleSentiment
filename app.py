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
    # Initialize text for webpage as blank
    user_input, topic_text, top_n_text, high_text, low_text, summary_text = "", "", "", "", "", ""
    
    if request.method == "POST":
        user_input = request.form["user_input"]

        # Get headlines on topic
        headlines = get_google_news_headlines(googlenews, user_input)

        # If there are headlines on the search topic
        if headlines:
            # Report sentiment for each headline
            sentiment_values = report_sentiment(sentimentizer, headlines)

            df = pd.DataFrame({'Sentiments': sentiment_values['sentiments'], 'Headlines': headlines})
            df.sort_values(by='Sentiments', inplace=True)

            # Summarize headlines
            summary = summarize_headlines(summarizer, df['Headlines'])

            # All text for webpage
            topic_text = 'Topic searched: "{}"'.format(user_input)

            top_n_text = '\nTop {} headlines pulled for search topic {}...'.format(len(headlines), user_input)

            high_text = '\nHighest rated sentiment of "{}" given to the following headline: \n"{}"'.format(sentiment_values['maxSentiment'], headlines[sentiment_values['maxSentimentIdx']])
            low_text = '\nLowest rated sentiment of "{}" given to the following headline: \n"{}"'.format(sentiment_values['minSentiment'], headlines[sentiment_values['minSentimentIdx']])
            
            summary_text = '\nSummary of all {} headlines: \n"{}"'.format(len(headlines), summary)

        # If no headlines are found
        else:
            topic_text = 'Topic searched: "{}"'.format(user_input)
            top_n_text = "No headlines found."
            high_text, low_text, summary_text = "", "", ""
    
    return render_template("index.html", 
                           topic_text = topic_text,
                           top_n_text = top_n_text,
                           high_text= high_text,
                           low_text = low_text,
                           summary_text = summary_text)


# Run the app if this script is executed
if __name__ == "__main__":
    app.run()
