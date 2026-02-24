import yfinance as yf
import logging
from textblob import TextBlob

logger = logging.getLogger(__name__)

def analyze_news_sentiment(ticker):
    """
    Fetch recent news for a ticker and calculate sentiment score (-1 to 1).
    Uses TextBlob for simple polarity analysis.
    """
    try:
        t = yf.Ticker(ticker)
        news = t.news
        
        if not news:
            return None
            
        total_score = 0
        count = 0
        headlines = []
        
        for item in news:
            title = item.get('title', '')
            if not title:
                continue
                
            # Calculate polarity: -1 (Negative) to +1 (Positive)
            blob = TextBlob(title)
            score = blob.sentiment.polarity
            
            total_score += score
            count += 1
            
            # Keep top 3 headlines for display
            if len(headlines) < 3:
                headlines.append(title)
                
        if count == 0:
            return None
            
        avg_score = total_score / count
        
        sentiment_label = 'Neutral'
        if avg_score > 0.1: sentiment_label = 'Positive'
        elif avg_score < -0.1: sentiment_label = 'Negative'
        
        return {
            'Score': round(avg_score, 2),
            'Label': sentiment_label,
            'Headlines': headlines
        }
        
    except Exception as e:
        logger.warning(f"Failed to analyze news for {ticker}: {e}")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ticker = 'TSLA'
    print(f"\nAnalyzing News for {ticker}...")
    news_sent = analyze_news_sentiment(ticker)
    print("News Sentiment:", news_sent)
