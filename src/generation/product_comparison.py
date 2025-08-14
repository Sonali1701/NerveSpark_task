from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import streamlit as st

@dataclass
class ComparisonFeature:
    """Class representing a feature in product comparison."""
    name: str
    values: Dict[str, Any]  # product_id -> value
    is_numeric: bool = False
    higher_is_better: bool = True

class ProductComparator:
    """Class for comparing multiple products based on their features."""
    
    def __init__(self, products: List[Dict[str, Any]]):
        """Initialize with a list of product dictionaries."""
        self.products = {p['product_id']: p for p in products}
        self.product_ids = list(self.products.keys())
    
    def extract_features(self) -> List[ComparisonFeature]:
        """Extract comparable features from products."""
        if not self.products:
            return []
        
        features = []
        
        # Add basic features
        basic_features = [
            ('price', 'Price', True, False),  # name, is_numeric, higher_is_better
            ('brand', 'Brand', False, None),
            ('category', 'Category', False, None),
        ]
        
        for key, name, is_numeric, higher_is_better in basic_features:
            values = {}
            for pid, product in self.products.items():
                values[pid] = product.get(key)
            features.append(ComparisonFeature(
                name=name,
                values=values,
                is_numeric=is_numeric,
                higher_is_better=higher_is_better
            ))
        
        # Add specification features
        spec_features = self._extract_specification_features()
        features.extend(spec_features)
        
        # Add review-based features
        review_features = self._extract_review_features()
        features.extend(review_features)
        
        return features
    
    def _extract_specification_features(self) -> List[ComparisonFeature]:
        """Extract features from product specifications."""
        if not self.products:
            return []
        
        # Collect all specification keys across products
        all_specs = set()
        for product in self.products.values():
            if 'specifications' in product:
                all_specs.update(product['specifications'].keys())
        
        features = []
        for spec_name in sorted(all_specs):
            values = {}
            is_numeric = True
            
            for pid, product in self.products.items():
                spec_value = product.get('specifications', {}).get(spec_name)
                if spec_value is not None:
                    # Try to convert to number if possible
                    try:
                        if any(c.isalpha() for c in str(spec_value)):
                            is_numeric = False
                        else:
                            spec_value = float(str(spec_value).replace(',', '').strip())
                    except (ValueError, AttributeError):
                        is_numeric = False
                values[pid] = spec_value
            
            # Determine if higher is better (for numeric specs)
            higher_is_better = True
            if spec_name.lower() in ['weight', 'dimensions', 'size']:
                higher_is_better = False
            
            features.append(ComparisonFeature(
                name=spec_name,
                values=values,
                is_numeric=is_numeric,
                higher_is_better=higher_is_better
            ))
        
        return features
    
    def _extract_review_features(self) -> List[ComparisonFeature]:
        """Extract features from product reviews."""
        if not self.products:
            return []
        
        features = []
        
        # Average rating
        avg_ratings = {}
        for pid, product in self.products.items():
            reviews = product.get('reviews', [])
            if reviews:
                avg_ratings[pid] = sum(r.get('rating', 0) for r in reviews) / len(reviews)
            else:
                avg_ratings[pid] = None
        
        if any(r is not None for r in avg_ratings.values()):
            features.append(ComparisonFeature(
                name="Average Rating",
                values=avg_ratings,
                is_numeric=True,
                higher_is_better=True
            ))
        
        # Review count
        review_counts = {}
        for pid, product in self.products.items():
            review_counts[pid] = len(product.get('reviews', []))
        
        features.append(ComparisonFeature(
            name="Number of Reviews",
            values=review_counts,
            is_numeric=True,
            higher_is_better=True
        ))
        
        # Sentiment (placeholder - would be implemented with actual sentiment analysis)
        # This is a simplified version
        sentiment_scores = {}
        for pid, product in self.products.items():
            reviews = product.get('reviews', [])
            if not reviews:
                sentiment_scores[pid] = None
                continue
                
            # Simple sentiment: count positive words (in a real app, use a proper sentiment analyzer)
            positive_words = {'good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'best'}
            negative_words = {'bad', 'poor', 'terrible', 'awful', 'hate', 'worst'}
            
            score = 0
            total_words = 0
            
            for review in reviews:
                text = review.get('text', '').lower()
                words = text.split()
                total_words += len(words)
                
                for word in words:
                    if word in positive_words:
                        score += 1
                    elif word in negative_words:
                        score -= 1
            
            sentiment_scores[pid] = score / max(1, total_words) if total_words > 0 else 0
        
        features.append(ComparisonFeature(
            name="Sentiment Score",
            values=sentiment_scores,
            is_numeric=True,
            higher_is_better=True
        ))
        
        return features
    
    def get_comparison_table(self) -> List[Dict[str, Any]]:
        """Generate a comparison table for the products."""
        features = self.extract_features()
        
        # Prepare table rows
        rows = []
        for feature in features:
            row = {'feature': feature.name}
            
            # Add values for each product
            for pid in self.product_ids:
                value = feature.values.get(pid)
                
                # Format the value
                if value is None:
                    display_value = "-"
                elif feature.is_numeric:
                    display_value = f"{value:,.2f}" if isinstance(value, (int, float)) else str(value)
                else:
                    display_value = str(value)
                
                row[pid] = display_value
            
            rows.append(row)
        
        return rows
    
    def get_recommendation(self, user_preferences: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Get a recommendation based on feature comparison.
        
        Args:
            user_preferences: Optional dictionary of feature weights. If None, all features are weighted equally.
            
        Returns:
            Dictionary mapping product IDs to recommendation scores.
        """
        features = self.extract_features()
        scores = {pid: 0.0 for pid in self.product_ids}
        
        # Default weights if not provided
        if user_preferences is None:
            user_preferences = {f.name: 1.0 for f in features}
        
        # Normalize weights
        total_weight = sum(abs(w) for w in user_preferences.values())
        if total_weight == 0:
            return {pid: 0.0 for pid in self.product_ids}
        
        weights = {k: v / total_weight for k, v in user_preferences.items()}
        
        # Calculate scores for each feature
        for feature in features:
            if feature.name not in weights or weights[feature.name] == 0:
                continue
                
            # Get values for this feature
            values = feature.values
            
            # Skip if no values
            if not any(v is not None for v in values.values()):
                continue
                
            # Normalize values
            valid_values = [v for v in values.values() if v is not None]
            
            if not valid_values:
                continue
                
            if feature.is_numeric:
                min_val = min(valid_values)
                max_val = max(valid_values)
                range_val = max_val - min_val
                
                if range_val == 0:
                    # All values are the same
                    normalized = {pid: 0.5 for pid in values}
                else:
                    normalized = {}
                    for pid, val in values.items():
                        if val is None:
                            normalized[pid] = 0
                        else:
                            # Normalize to [0, 1]
                            norm_val = (val - min_val) / range_val
                            if not feature.higher_is_better:
                                norm_val = 1 - norm_val
                            normalized[pid] = norm_val
            else:
                # For non-numeric, just use 1 for match, 0 for non-match
                # This is a simplification - in a real app, you might want more sophisticated comparison
                normalized = {}
                for pid, val in values.items():
                    normalized[pid] = 1 if val is not None else 0
            
            # Add weighted score to each product
            weight = weights.get(feature.name, 0)
            for pid, score in normalized.items():
                if pid in scores:
                    scores[pid] += score * weight
        
        return scores
