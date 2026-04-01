import os
from newsapi import NewsApiClient
from textblob import TextBlob
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta

# Load your API key from the .env file
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")


def get_headlines(ticker, company_name=None):
    """
    Fetches recent news headlines for a given stock ticker.
    We search by both the ticker and company name for better results.
    """
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)

    # Search term — e.g. "AAPL OR Apple Inc"
    query = ticker
    if company_name:
        query = f"{ticker} OR {company_name}"

    # Get headlines from the last 7 days
    week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    today = datetime.now().strftime('%Y-%m-%d')

    response = newsapi.get_everything(
        q=query,
        from_param=week_ago,
        to=today,
        language='en',
        sort_by='relevancy',
        page_size=20         # grab 20 headlines max
    )

    articles = response.get('articles', [])
    return articles


def analyze_sentiment(articles):
    """
    Takes a list of news articles and scores each headline
    using TextBlob sentiment analysis.

    TextBlob gives each piece of text two scores:
    - Polarity: -1.0 (very negative) to +1.0 (very positive)
    - Subjectivity: 0.0 (objective fact) to 1.0 (personal opinion)
    """
    results = []

    for article in articles:
        headline = article.get('title', '')
        published = article.get('publishedAt', '')[:10]  # just the date part

        if not headline or headline == '[Removed]':
            continue

        # Run sentiment analysis on the headline
        blob = TextBlob(headline)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Label it human-readably
        if polarity > 0.1:
            label = "🟢 Positive"
        elif polarity < -0.1:
            label = "🔴 Negative"
        else:
            label = "🟡 Neutral"

        results.append({
            'Date': published,
            'Headline': headline,
            'Sentiment': label,
            'Polarity': round(polarity, 3),
            'Subjectivity': round(subjectivity, 3)
        })

    return pd.DataFrame(results)


def get_overall_sentiment(df):
    """
    Averages all the polarity scores to give one
    overall sentiment reading for the stock.
    """
    if df.empty:
        return 0, "🟡 Neutral"

    avg_polarity = df['Polarity'].mean()

    if avg_polarity > 0.1:
        overall = "🟢 Positive"
    elif avg_polarity < -0.1:
        overall = "🔴 Negative"
    else:
        overall = "🟡 Neutral"

    return round(avg_polarity, 3), overall