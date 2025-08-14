import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import json
import os
from pathlib import Path

@dataclass
class UserPreferenceModel:
    """Class to model user preferences based on interactions."""
    user_id: str
    feature_weights: Dict[str, float] = field(default_factory=dict)
    viewed_products: Set[str] = field(default_factory=set)
    purchased_products: Set[str] = field(default_factory=set)
    wishlisted_products: Set[str] = field(default_factory=set)
    search_history: List[Dict] = field(default_factory=list)
    
    def update_from_interaction(self, interaction_type: str, product_id: str, **kwargs):
        """Update user preferences based on interaction type."""
        if interaction_type == 'view':
            self.viewed_products.add(product_id)
        elif interaction_type == 'purchase':
            self.purchased_products.add(product_id)
        elif interaction_type == 'wishlist':
            self.wishlisted_products.add(product_id)
        elif interaction_type == 'search':
            self.search_history.append({
                'query': kwargs.get('query', ''),
                'filters': kwargs.get('filters', {}),
                'timestamp': kwargs.get('timestamp')
            })
    
    def calculate_implicit_feedback(self, product_features: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate implicit feedback scores for product features."""
        if not (self.viewed_products or self.purchased_products or self.wishlisted_products):
            return {}
        
        # Collect all features from interacted products
        all_features = set()
        for product_id in self.viewed_products | self.purchased_products | self.wishlisted_products:
            if product_id in product_features:
                all_features.update(product_features[product_id].keys())
        
        # Initialize feature importance
        feature_importance = {feature: 0.0 for feature in all_features}
        
        # Weights for different interaction types
        interaction_weights = {
            'purchase': 2.0,
            'wishlist': 1.5,
            'view': 1.0
        }
        
        # Update importance based on interactions
        for product_id in self.viewed_products:
            if product_id in product_features:
                for feature, value in product_features[product_id].items():
                    if isinstance(value, (int, float)):
                        feature_importance[feature] += interaction_weights['view'] * abs(value)
                    else:
                        feature_importance[feature] += interaction_weights['view']
        
        for product_id in self.wishlisted_products:
            if product_id in product_features:
                for feature, value in product_features[product_id].items():
                    if isinstance(value, (int, float)):
                        feature_importance[feature] += interaction_weights['wishlist'] * abs(value)
                    else:
                        feature_importance[feature] += interaction_weights['wishlist']
        
        for product_id in self.purchased_products:
            if product_id in product_features:
                for feature, value in product_features[product_id].items():
                    if isinstance(value, (int, float)):
                        feature_importance[feature] += interaction_weights['purchase'] * abs(value)
                    else:
                        feature_importance[feature] += interaction_weights['purchase']
        
        # Normalize importance scores
        if feature_importance:
            max_importance = max(feature_importance.values())
            if max_importance > 0:
                feature_importance = {k: v / max_importance for k, v in feature_importance.items()}
        
        return feature_importance
    
    def get_preference_weights(self, product_features: Dict[str, Dict]) -> Dict[str, float]:
        """Get preference weights for recommendation.
        
        Args:
            product_features: Dictionary mapping product IDs to their feature dictionaries
            
        Returns:
            Dictionary mapping feature names to preference weights
        """
        # Start with existing weights
        weights = self.feature_weights.copy()
        
        # Update with implicit feedback
        implicit_weights = self.calculate_implicit_feedback(product_features)
        
        # Combine explicit and implicit weights
        for feature, importance in implicit_weights.items():
            if feature in weights:
                # Average explicit and implicit weights
                weights[feature] = (weights[feature] + importance) / 2
            else:
                weights[feature] = importance
        
        return weights
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'user_id': self.user_id,
            'feature_weights': self.feature_weights,
            'viewed_products': list(self.viewed_products),
            'purchased_products': list(self.purchased_products),
            'wishlisted_products': list(self.wishlisted_products),
            'search_history': self.search_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserPreferenceModel':
        """Create from dictionary."""
        model = cls(user_id=data['user_id'])
        model.feature_weights = data.get('feature_weights', {})
        model.viewed_products = set(data.get('viewed_products', []))
        model.purchased_products = set(data.get('purchased_products', []))
        model.wishlisted_products = set(data.get('wishlisted_products', []))
        model.search_history = data.get('search_history', [])
        return model


class PreferenceLearner:
    """Class to manage user preference learning across multiple users."""
    
    def __init__(self, data_dir: str = 'data/user_preferences'):
        """Initialize the preference learner.
        
        Args:
            data_dir: Directory to store user preference data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.user_models: Dict[str, UserPreferenceModel] = {}
        self.load_all_models()
    
    def get_user_model(self, user_id: str) -> UserPreferenceModel:
        """Get or create a user preference model."""
        if user_id not in self.user_models:
            self.user_models[user_id] = UserPreferenceModel(user_id=user_id)
        return self.user_models[user_id]
    
    def update_user_preferences(
        self,
        user_id: str,
        interaction_type: str,
        product_id: str,
        **kwargs
    ) -> UserPreferenceModel:
        """Update user preferences based on interaction."""
        user_model = self.get_user_model(user_id)
        user_model.update_from_interaction(interaction_type, product_id, **kwargs)
        self.save_user_model(user_id)
        return user_model
    
    def get_user_preference_weights(
        self,
        user_id: str,
        product_features: Dict[str, Dict]
    ) -> Dict[str, float]:
        """Get preference weights for a user."""
        user_model = self.get_user_model(user_id)
        return user_model.get_preference_weights(product_features)
    
    def save_user_model(self, user_id: str):
        """Save a user's preference model to disk."""
        if user_id not in self.user_models:
            return
            
        user_file = self.data_dir / f"{user_id}.json"
        with open(user_file, 'w') as f:
            json.dump(self.user_models[user_id].to_dict(), f, indent=2)
    
    def load_user_model(self, user_id: str) -> Optional[UserPreferenceModel]:
        """Load a user's preference model from disk."""
        user_file = self.data_dir / f"{user_id}.json"
        if not user_file.exists():
            return None
            
        try:
            with open(user_file, 'r') as f:
                data = json.load(f)
                return UserPreferenceModel.from_dict(data)
        except (json.JSONDecodeError, IOError):
            return None
    
    def load_all_models(self):
        """Load all user preference models from disk."""
        self.user_models = {}
        for user_file in self.data_dir.glob('*.json'):
            user_id = user_file.stem
            model = self.load_user_model(user_id)
            if model:
                self.user_models[user_id] = model
    
    def get_or_create_user_model(self, user_id: str) -> UserPreferenceModel:
        """Get a user's preference model, creating it if it doesn't exist."""
        if user_id not in self.user_models:
            model = self.load_user_model(user_id)
            if model is None:
                model = UserPreferenceModel(user_id=user_id)
            self.user_models[user_id] = model
        return self.user_models[user_id]
    
    def update_feature_weights(
        self,
        user_id: str,
        feature_weights: Dict[str, float]
    ) -> UserPreferenceModel:
        """Update a user's explicit feature weights."""
        user_model = self.get_or_create_user_model(user_id)
        user_model.feature_weights.update(feature_weights)
        self.save_user_model(user_id)
        return user_model
    
    def get_user_activity_summary(self, user_id: str) -> Dict[str, int]:
        """Get a summary of a user's activity."""
        user_model = self.get_or_create_user_model(user_id)
        return {
            'views': len(user_model.viewed_products),
            'purchases': len(user_model.purchased_products),
            'wishlist': len(user_model.wishlisted_products),
            'searches': len(user_model.search_history)
        }
