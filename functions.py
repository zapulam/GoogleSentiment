from GoogleNews import GoogleNews
from transformers import pipeline
from transformers import logging


# Function to fetch headlines based on a user-specified prompt
def get_google_news_headlines(googlenews, prompt):
    try:
        googlenews.get_news(prompt)
        articles = googlenews.results()

        headlines = []
        for article in articles:
            headline = article['title']
            headlines.append(headline)

        return headlines

    except Exception as e:
        return str(e)


# Returns avergae sentiment score of all headlines
def report_sentiment(sentimentizer, headlines):
    try:
        sentiments = sentimentizer(headlines)

        for i, sentiment in enumerate(sentiments):
            if sentiment['label'] == 'NEGATIVE':
                sentiments[i]['score'] = sentiments[i]['score'] * -1

        avg_sentiment = sum(sentiment['score'] for sentiment in sentiments) / len(sentiments)

        maxSentiment = round(max(sentiments, key=lambda x:x['score'])['score'], 5)
        maxSentimentIdx = max(range(len(sentiments)), key=lambda i: sentiments[i].get('score', 0))

        minSentiment = round(min(sentiments, key=lambda x:x['score'])['score'], 5)
        minSentimentIdx = min(range(len(sentiments)), key=lambda i: sentiments[i].get('score', 0))

        return sentiments, avg_sentiment, maxSentiment, maxSentimentIdx, minSentiment, minSentimentIdx

    except Exception as e:
        return str(e)


# Summarize headlines
def summarize_headlines(summarizer, headlines):
    try:
        headlines = '. '.join(headlines.tolist()[:512])
        tokens = headlines.split()
        first_N_tokens = ' '.join(tokens[:512])

        summary = summarizer(first_N_tokens, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        return summary
    
    except Exception as e:
        return str(e)
