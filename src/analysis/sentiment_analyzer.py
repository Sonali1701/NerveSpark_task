from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

@dataclass
class ReviewSentiment:
    """Class to hold review sentiment analysis results."""
    review_text: str
    sentiment: str  # 'positive', 'negative', 'neutral'
    confidence: float
    aspects: Dict[str, Dict[str, float]] = None  # aspect -> {'sentiment': str, 'confidence': float}

class SentimentAnalyzer:
    """Class for analyzing sentiment of product reviews."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """Initialize the sentiment analyzer with a pre-trained model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.nlp = None
        self.aspects = [
            'quality', 'price', 'performance', 'design', 'battery',
            'camera', 'screen', 'sound', 'ease of use', 'features'
        ]
    
    def load_model(self):
        """Load the sentiment analysis model."""
        if self.model is None or self.tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.nlp = pipeline(
                    'sentiment-analysis',
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1
                )
            except Exception as e:
                print(f"Error loading model: {e}")
                # Fallback to a simpler model
                self.nlp = pipeline('sentiment-analysis')
    
    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """Analyze the sentiment of a given text.
        
        Args:
            text: The text to analyze
            
        Returns:
            A tuple of (sentiment_label, confidence_score)
        """
        self.load_model()
        
        try:
            result = self.nlp(text)[0]
            return result['label'], result['score']
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            # Fallback to a simple rule-based approach
            return self._rule_based_sentiment(text)
    
    def analyze_review(self, review_text: str) -> ReviewSentiment:
        """Analyze a product review for overall sentiment and aspect-based sentiment.
        
        Args:
            review_text: The review text to analyze
            
        Returns:
            ReviewSentiment object with analysis results
        """
        # Get overall sentiment
        sentiment_label, confidence = self.analyze_sentiment(review_text)
        
        # Initialize result
        result = ReviewSentiment(
            review_text=review_text,
            sentiment=sentiment_label.lower(),
            confidence=confidence
        )
        
        # Analyze aspects if possible
        try:
            result.aspects = self._analyze_aspects(review_text)
        except Exception as e:
            print(f"Error in aspect-based analysis: {e}")
            result.aspects = {}
        
        return result
    
    def _analyze_aspects(self, text: str) -> Dict[str, Dict[str, float]]:
        """Analyze sentiment for different aspects mentioned in the text."""
        aspect_sentiments = {}
        
        # Simple approach: Look for aspect mentions and analyze surrounding text
        for aspect in self.aspects:
            if aspect.lower() in text.lower():
                # Find the sentence containing the aspect
                sentences = text.split('.')
                for sent in sentences:
                    if aspect.lower() in sent.lower():
                        # Analyze the sentence containing the aspect
                        sentiment, confidence = self.analyze_sentiment(sent)
                        aspect_sentiments[aspect] = {
                            'sentiment': sentiment.lower(),
                            'confidence': confidence
                        }
                        break
        
        return aspect_sentiments
    
    def _rule_based_sentiment(self, text: str) -> Tuple[str, float]:
        """Simple rule-based sentiment analysis as a fallback."""
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'best',
            'awesome', 'fantastic', 'wonderful', 'outstanding', 'superb', 'brilliant'
        }
        
        negative_words = {
            'bad', 'poor', 'terrible', 'awful', 'hate', 'worst', 'disappointing',
            'disappointed', 'waste', 'rubbish', 'garbage', 'trash', 'useless'
        }
        
        # Count positive and negative words
        words = text.lower().split()
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        
        # Determine sentiment
        if pos_count > neg_count:
            return "POSITIVE", min(0.99, 0.5 + (pos_count * 0.1))
        elif neg_count > pos_count:
            return "NEGATIVE", min(0.99, 0.5 + (neg_count * 0.1))
        else:
            return "NEUTRAL", 0.5
    
    def get_sentiment_summary(self, reviews: List[Dict]) -> Dict[str, Any]:
        """Get a summary of sentiment across multiple reviews.
        
        Args:
            reviews: List of review dictionaries with 'text' and optionally 'rating' keys
            
        Returns:
            Dictionary with sentiment summary statistics
        """
        if not reviews:
            return {
                'average_rating': 0,
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
                'aspect_sentiments': {},
                'total_reviews': 0
            }
        
        # Initialize counters
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        aspect_scores = {aspect: [] for aspect in self.aspects}
        total_rating = 0
        
        # Analyze each review
        for review in reviews:
            text = review.get('text', '')
            rating = review.get('rating', 0)
            
            # Skip empty reviews
            if not text.strip():
                continue
                
            # Analyze sentiment
            analysis = self.analyze_review(text)
            sentiment = analysis.sentiment.lower()
            
            # Update sentiment counts
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
            else:
                sentiment_counts['neutral'] += 1
            
            # Update aspect scores
            if analysis.aspects:
                for aspect, data in analysis.aspects.items():
                    if aspect in aspect_scores:
                        # Convert sentiment to numerical score (-1 to 1)
                        if data['sentiment'] == 'positive':
                            score = data['confidence']
                        elif data['sentiment'] == 'negative':
                            score = -data['confidence']
                        else:
                            score = 0
                        aspect_scores[aspect].append(score)
            
            # Add rating if available
            if rating > 0:
                total_rating += rating
        
        # Calculate average rating
        valid_ratings = sum(1 for r in reviews if r.get('rating', 0) > 0)
        avg_rating = total_rating / valid_ratings if valid_ratings > 0 else 0
        
        # Calculate aspect sentiment scores
        aspect_sentiments = {}
        for aspect, scores in aspect_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                aspect_sentiments[aspect] = {
                    'score': avg_score,
                    'count': len(scores)
                }
        
        return {
            'average_rating': round(avg_rating, 1) if avg_rating > 0 else 0,
            'sentiment_distribution': sentiment_counts,
            'aspect_sentiments': aspect_sentiments,
            'total_reviews': len(reviews)
        }
