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

    except Exception as e:
        return str(e)
    
    return headlines


# Returns avergae sentiment score of all headlines
def report_sentiment(headlines):
    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model ='distilbert-base-uncased-finetuned-sst-2-english')
        sentiments = sentiment_pipeline(headlines)

        for i, sentiment in enumerate(sentiments):
            if sentiment['label'] == 'NEGATIVE':
                sentiments[i]['score'] = sentiments[i]['score'] * -1

    except Exception as e:
        return str(e)
    
    return sentiments, sum(sentiment['score'] for sentiment in sentiments) / len(sentiments)


# Main function
def main():
    try:
        googlenews = GoogleNews()
        googlenews.enableException(True)

        print('\nGoogleNews Version: ', googlenews.getVersion())
    except Exception as e:
        return str(e)

    user_prompt = input("\nEnter a search prompt: ")
    # start = input("\nEnter a start date (mm/dd/yyyy): ")
    # end = input("\nEnter an end date (mm/dd/yyyy): ")

    # try:
    #     googlenews = GoogleNews(start=start, end=end)
    # except Exception as e:
    #     return str(e)

    headlines = get_google_news_headlines(googlenews, user_prompt)

    if headlines:
        print("\nTop {} headlines pulled...".format(len(headlines)))

        sentiments, avg_sentiment = report_sentiment(headlines)

        maxSentiment = round(max(sentiments, key=lambda x:x['score'])['score'], 5)
        maxSentimentIdx = max(range(len(sentiments)), key=lambda i: sentiments[i].get('score', 0))

        minSentiment = round(min(sentiments, key=lambda x:x['score'])['score'], 5)
        minSentimentIdx = min(range(len(sentiments)), key=lambda i: sentiments[i].get('score', 0))

        print('\nHighest rated sentiment of "{}" given to the following headline: \n"{}"'.format(maxSentiment, headlines[maxSentimentIdx]))
        print('\nLowest rated sentiment of "{}" given to the following headline: \n"{}"'.format(minSentiment, headlines[minSentimentIdx]))

    else:
        print("No headlines found.")


if __name__ == "__main__":
    main()
