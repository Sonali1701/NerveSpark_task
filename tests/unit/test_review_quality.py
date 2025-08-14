import pytest
from src.analysis.review_quality import ReviewQualityAssessor, ReviewQualityScore

def test_review_quality_assessor_initialization():
    """Test that the ReviewQualityAssessor initializes correctly."""
    assessor = ReviewQualityAssessor()
    assert assessor is not None
    assert hasattr(assessor, 'spam_indicators')
    assert hasattr(assessor, 'stop_words')

def test_assess_review_quality_short():
    """Test quality assessment for a very short review."""
    assessor = ReviewQualityAssessor()
    review_text = "Great!"
    result = assessor.assess_review(review_text, "Sample Product")
    
    assert isinstance(result, ReviewQualityScore)
    assert "review_too_short" in result.flags
    assert 0 <= result.authenticity_score <= 1
    assert 0 <= result.relevance_score <= 1

def test_assess_review_quality_spam():
    """Test quality assessment for a spam-like review."""
    assessor = ReviewQualityAssessor()
    review_text = "BUY NOW! LIMITED OFFER! Click here: http://spam.com"
    result = assessor.assess_review(review_text, "Sample Product")
    
    assert any("spam_pattern" in flag for flag in result.flags)
    assert result.authenticity_score < 0.5  # Should be penalized for spam indicators

def test_assess_review_quality_good():
    """Test quality assessment for a good quality review."""
    assessor = ReviewQualityAssessor()
    review_text = """
    I've been using this product for a month now and I'm very satisfied. 
    The build quality is excellent and it works as advertised. 
    The battery life could be better, but overall it's a great purchase.
    """
    result = assessor.assess_review(review_text, "Sample Product")
    
    assert len(result.flags) == 0  # No quality flags
    assert result.authenticity_score >= 0.7  # Should score high on authenticity
    assert result.relevance_score >= 0.5  # Should be relevant to the product

@pytest.mark.parametrize("review_text,expected_score", [
    ("This is a great product!", 0.7),  # Positive sentiment
    ("I hate this product!", 0.3),      # Negative sentiment
    ("It's okay, not great", 0.5),      # Neutral sentiment
])
def test_sentiment_impact(review_text, expected_score):
    """Test how sentiment affects the quality score."""
    assessor = ReviewQualityAssessor()
    result = assessor.assess_review(review_text, "Sample Product")
    
    # We're not testing exact values, just the general trend
    if expected_score > 0.6:
        assert result.authenticity_score > 0.6
    elif expected_score < 0.4:
        assert result.authenticity_score < 0.4
    else:
        assert 0.4 <= result.authenticity_score <= 0.6
