import pytest
from src.analysis.sentiment_analyzer import SentimentAnalyzer, ReviewSentiment

def test_sentiment_analyzer_initialization():
    """Test that the SentimentAnalyzer initializes correctly."""
    analyzer = SentimentAnalyzer()
    assert analyzer is not None
    assert hasattr(analyzer, 'analyzer')

def test_analyze_review_basic():
    """Test basic sentiment analysis on a simple review."""
    analyzer = SentimentAnalyzer()
    review_text = "I love this product! It's amazing."
    
    result = analyzer.analyze_review(review_text)
    
    assert isinstance(result, ReviewSentiment)
    assert hasattr(result, 'sentiment')
    assert hasattr(result, 'confidence')
    assert hasattr(result, 'aspects')
    assert result.sentiment in ['positive', 'negative', 'neutral']
    assert 0 <= result.confidence <= 1.0

def test_aspect_based_analysis():
    """Test aspect-based sentiment analysis."""
    analyzer = SentimentAnalyzer()
    review_text = "The camera quality is excellent but the battery life is terrible."
    
    result = analyzer.analyze_review(review_text)
    
    assert 'camera' in result.aspects
    assert 'battery' in result.aspects
    assert result.aspects['camera']['sentiment'] == 'positive'
    assert result.aspects['battery']['sentiment'] == 'negative'

@pytest.mark.parametrize("text,expected_sentiment", [
    ("This is amazing!", "positive"),
    ("I hate this product!", "negative"),
    ("It's okay, I guess.", "neutral"),
])
def test_sentiment_classification(text, expected_sentiment):
    """Test sentiment classification for different types of text."""
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze_review(text)
    assert result.sentiment == expected_sentiment

def test_get_sentiment_summary():
    """Test generating a sentiment summary from multiple reviews."""
    analyzer = SentimentAnalyzer()
    reviews = [
        {"text": "Great product!", "rating": 5},
        {"text": "Not bad, could be better", "rating": 3},
        {"text": "Terrible experience", "rating": 1},
    ]
    
    summary = analyzer.get_sentiment_summary(reviews)
    
    assert 'total_reviews' in summary
    assert 'average_rating' in summary
    assert 'sentiment_distribution' in summary
    assert 'aspect_sentiments' in summary
    
    assert summary['total_reviews'] == 3
    assert 1 <= summary['average_rating'] <= 5
    assert 'positive' in summary['sentiment_distribution']
    assert 'negative' in summary['sentiment_distribution']

def test_fallback_analysis():
    """Test that the fallback analysis works when the main analyzer fails."""
    # Create a mock analyzer that will raise an exception
    class FailingAnalyzer:
        def __call__(self, *args, **kwargs):
            raise Exception("Analyzer failed")
    
    # Replace the analyzer with our failing one
    original_analyzer = SentimentAnalyzer.analyzer
    SentimentAnalyzer.analyzer = FailingAnalyzer()
    
    try:
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_review("This is a test review.")
        
        # Should still return a valid result using the fallback
        assert result is not None
        assert hasattr(result, 'sentiment')
        assert hasattr(result, 'confidence')
    finally:
        # Restore the original analyzer
        SentimentAnalyzer.analyzer = original_analyzer
