from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import json

@dataclass
class ProductRecommendation:
    """Class representing a product recommendation result."""
    product_id: str
    title: str
    score: float
    explanation: str
    metadata: Dict[str, Any]

class RecommendationEngine:
    """Main recommendation engine for the RAG system."""
    
    def __init__(self, vector_store):
        """Initialize the recommendation engine with a vector store.
        
        Args:
            vector_store: Initialized VectorStore instance
        """
        self.vector_store = vector_store
        self.aspect_weights = {
            'title': 0.3,
            'description': 0.2,
            'specifications': 0.3,
            'reviews': 0.2
        }
    
    def search_products(
        self, 
        query: str,
        n_results: int = 5,
        aspect_weights: Optional[Dict[str, float]] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[ProductRecommendation]:
        """Search for products based on a query with weighted aspects.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            aspect_weights: Optional weights for different aspects
            filter_conditions: Optional filters to apply
            
        Returns:
            List of product recommendations
        """
        if aspect_weights is None:
            aspect_weights = self.aspect_weights
            
        # Normalize weights
        total_weight = sum(aspect_weights.values())
        if total_weight <= 0:
            raise ValueError("Sum of aspect weights must be greater than 0")
            
        weights = {k: v/total_weight for k, v in aspect_weights.items()}
        
        # Search each aspect
        results = {}
        for aspect, weight in weights.items():
            if weight > 0:
                aspect_results = self.vector_store.search(
                    query=query,
                    aspect=aspect,
                    n_results=n_results * 2,  # Get more results to have enough for combination
                    filter_conditions=filter_conditions
                )
                
                # Apply weights and accumulate scores
                for result in aspect_results:
                    product_id = result['id']
                    if product_id not in results:
                        results[product_id] = {
                            'id': product_id,
                            'title': result['metadata'].get('title', ''),
                            'scores': {},
                            'metadata': result['metadata']
                        }
                    results[product_id]['scores'][aspect] = result['score'] * weight
        
        # Combine scores
        combined_results = []
        for product_id, data in results.items():
            total_score = sum(data['scores'].values())
            # Generate explanation
            aspect_scores = [
                f"{aspect}: {score:.2f}" 
                for aspect, score in data['scores'].items()
            ]
            explanation = f"Matched on: {', '.join(aspect_scores)}"
            
            combined_results.append(ProductRecommendation(
                product_id=product_id,
                title=data['title'],
                score=total_score,
                explanation=explanation,
                metadata=data['metadata']
            ))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results[:n_results]
    
    def get_similar_products(
        self, 
        product_id: str,
        n_results: int = 5,
        aspect_weights: Optional[Dict[str, float]] = None
    ) -> List[ProductRecommendation]:
        """Get similar products to a given product ID.
        
        Args:
            product_id: ID of the product to find similar items for
            n_results: Number of similar products to return
            aspect_weights: Optional weights for different aspects
            
        Returns:
            List of similar product recommendations
        """
        if aspect_weights is None:
            aspect_weights = self.aspect_weights
            
        # Normalize weights
        total_weight = sum(aspect_weights.values())
        if total_weight <= 0:
            raise ValueError("Sum of aspect weights must be greater than 0")
            
        weights = {k: v/total_weight for k, v in aspect_weights.items()}
        
        # Get similar products for each aspect
        results = {}
        for aspect, weight in weights.items():
            if weight > 0:
                aspect_results = self.vector_store.get_similar_products(
                    product_id=product_id,
                    aspect=aspect,
                    n_results=n_results * 2  # Get more results to have enough for combination
                )
                
                # Apply weights and accumulate scores
                for result in aspect_results:
                    similar_id = result['id']
                    if similar_id not in results:
                        results[similar_id] = {
                            'id': similar_id,
                            'title': result['metadata'].get('title', ''),
                            'scores': {},
                            'metadata': result['metadata']
                        }
                    results[similar_id]['scores'][aspect] = result['score'] * weight
        
        # Combine scores
        combined_results = []
        for product_id, data in results.items():
            total_score = sum(data['scores'].values())
            # Generate explanation
            aspect_scores = [
                f"{aspect}: {score:.2f}" 
                for aspect, score in data['scores'].items()
            ]
            explanation = f"Similar on: {', '.join(aspect_scores)}"
            
            combined_results.append(ProductRecommendation(
                product_id=product_id,
                title=data['title'],
                score=total_score,
                explanation=explanation,
                metadata=data['metadata']
            ))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results[:n_results]
    
    def get_personalized_recommendations(
        self,
        user_history: List[Dict[str, Any]],
        n_results: int = 5,
        aspect_weights: Optional[Dict[str, float]] = None
    ) -> List[ProductRecommendation]:
        """Get personalized recommendations based on user history.
        
        Args:
            user_history: List of user interactions (e.g., views, purchases, ratings)
            n_results: Number of recommendations to return
            aspect_weights: Optional weights for different aspects
            
        Returns:
            List of personalized product recommendations
        """
        if not user_history:
            return []
            
        if aspect_weights is None:
            aspect_weights = self.aspect_weights
            
        # Get similar products for each item in user history
        all_recommendations = {}
        
        for interaction in user_history:
            product_id = interaction.get('product_id')
            if not product_id:
                continue
                
            # Get similar products to this item
            similar = self.get_similar_products(
                product_id=product_id,
                n_results=n_results * 2,  # Get more to have enough for combination
                aspect_weights=aspect_weights
            )
            
            # Apply interaction weight (e.g., higher for purchases, lower for views)
            weight = {
                'purchase': 1.0,
                'rating': 0.8,
                'view': 0.5
            }.get(interaction.get('type', 'view'), 0.5)
            
            # Apply rating if available
            if 'rating' in interaction:
                weight *= (interaction['rating'] / 5.0)  # Normalize to 0-1
            
            # Accumulate scores
            for rec in similar:
                if rec.product_id not in all_recommendations:
                    all_recommendations[rec.product_id] = {
                        'product_id': rec.product_id,
                        'title': rec.title,
                        'score': 0.0,
                        'metadata': rec.metadata,
                        'sources': []
                    }
                all_recommendations[rec.product_id]['score'] += rec.score * weight
                all_recommendations[rec.product_id]['sources'].append({
                    'source_product': product_id,
                    'score': rec.score,
                    'interaction': interaction.get('type', 'view')
                })
        
        # Convert to list and sort by score
        recommendations = list(all_recommendations.values())
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Format as ProductRecommendation objects
        result = []
        for i, rec in enumerate(recommendations[:n_results]):
            # Generate explanation
            sources = ", ".join(f"{s['source_product']} ({s['interaction']})" 
                               for s in rec['sources'][:2])
            if len(rec['sources']) > 2:
                sources += f" and {len(rec['sources']) - 2} more"
                
            result.append(ProductRecommendation(
                product_id=rec['product_id'],
                title=rec['title'],
                score=rec['score'],
                explanation=f"Recommended based on: {sources}",
                metadata=rec['metadata']
            ))
            
        return result
