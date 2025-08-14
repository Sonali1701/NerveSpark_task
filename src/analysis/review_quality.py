import re
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from datetime import datetime, timedelta

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

@dataclass
class ReviewQualityScore:
    """Class to hold review quality assessment results."""
    authenticity_score: float  # 0-1 scale
    relevance_score: float  # 0-1 scale
    flags: List[str]  # Quality flags

class ReviewQualityAssessor:
    """Assess review quality, authenticity, and relevance."""
    
    def __init__(self):
        self.spam_indicators = [
            r'http[s]?://\S+',  # URLs
            r'[A-Z]{3,}',  # Excessive caps
            r'!{3,}|\?{3,}|\${3,}',  # Repeated punctuation
            r'\b(?:free|win|prize|guarantee|limited|offer|discount)\b',
        ]
        self.stop_words = set(stopwords.words('english'))
    
    def assess_review(self, review_text: str, product_title: str = "") -> ReviewQualityScore:
        """Assess a single review's quality."""
        flags = []
        
        # Check for spam indicators
        for pattern in self.spam_indicators:
            if re.search(pattern, review_text, re.IGNORECASE):
                flags.append(f"spam_pattern: {pattern[:20]}...")
        
        # Check review length
        words = word_tokenize(review_text)
        if len(words) < 10:
            flags.append("review_too_short")
        
        # Check for relevance to product
        relevance = self._calculate_relevance(review_text, product_title)
        
        # Calculate authenticity score (1.0 is best)
        auth_score = max(0, 1.0 - (len(flags) * 0.15))
        
        return ReviewQualityScore(
            authenticity_score=auth_score,
            relevance_score=relevance,
            flags=flags
        )
    
    def _calculate_relevance(self, text: str, product_title: str) -> float:
        """Calculate how relevant the review is to the product."""
        if not product_title:
            return 0.7  # Default if no title to compare with
            
        # Simple word overlap between review and product title
        title_words = set(word.lower() for word in word_tokenize(product_title) 
                         if word.lower() not in self.stop_words)
        review_words = set(word.lower() for word in word_tokenize(text) 
                          if word.lower() not in self.stop_words)
        
        if not title_words:
            return 0.5
            
        overlap = len(title_words & review_words) / len(title_words)
        return min(1.0, overlap * 1.5)  # Cap at 1.0
    
    def batch_assess(self, reviews: List[Dict]) -> Dict[str, ReviewQualityScore]:
        """Assess multiple reviews at once."""
        return {
            str(i): self.assess_review(
                review.get('text', ''),
                review.get('product_title', '')
            )
            for i, review in enumerate(reviews)
        }
