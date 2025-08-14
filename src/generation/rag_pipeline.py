from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import numpy as np
from ..retrieval.vector_store import VectorStore
from ..embedding.embedding_generator import EmbeddingGenerator
from ..analysis.sentiment_analyzer import SentimentAnalyzer
from ..analysis.review_quality import ReviewQualityAssessor

@dataclass
class RAGResponse:
    """Container for RAG pipeline response."""
    response_text: str
    context: Dict[str, Any]
    source_documents: List[Dict]
    metadata: Dict[str, Any] = None

class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for product recommendations."""
    
    def __init__(self, 
                 vector_store: VectorStore,
                 embedding_generator: EmbeddingGenerator,
                 sentiment_analyzer: SentimentAnalyzer = None,
                 quality_assessor: ReviewQualityAssessor = None):
        """Initialize the RAG pipeline."""
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer()
        self.quality_assessor = quality_assessor or ReviewQualityAssessor()
    
    def generate_recommendation(self, 
                              query: str, 
                              user_history: List[Dict] = None,
                              num_results: int = 3) -> RAGResponse:
        """Generate a product recommendation using RAG.
        
        Args:
            query: User's query or request for a recommendation
            user_history: List of user's past interactions
            num_results: Number of results to return
            
        Returns:
            RAGResponse with the generated recommendation and context
        """
        # Step 1: Retrieve relevant products
        retrieved_docs = self.vector_store.search(
            query=query,
            n_results=num_results * 2  # Retrieve more for filtering
        )
        
        # Step 2: Process and filter retrieved documents
        processed_docs = self._process_retrieved_docs(retrieved_docs)
        
        # Step 3: Apply personalization from user history
        if user_history:
            processed_docs = self._apply_personalization(processed_docs, user_history)
        
        # Step 4: Select top results
        top_docs = self._select_top_results(processed_docs, num_results)
        
        # Step 5: Generate response
        response_text = self._generate_response(query, top_docs)
        
        return RAGResponse(
            response_text=response_text,
            context={
                'query': query,
                'num_retrieved': len(retrieved_docs),
                'num_filtered': len(top_docs)
            },
            source_documents=top_docs,
            metadata={
                'model': 'RAG-v1',
                'timestamp': str(np.datetime64('now'))
            }
        )
    
    def _process_retrieved_docs(self, docs: List[Dict]) -> List[Dict]:
        """Process and enhance retrieved documents with additional metadata."""
        processed = []
        
        for doc in docs:
            # Skip if already processed
            if 'processed' in doc:
                processed.append(doc)
                continue
                
            # Analyze sentiment of reviews if present
            if 'reviews' in doc and self.sentiment_analyzer:
                reviews = doc['reviews']
                if reviews and len(reviews) > 0:
                    sentiment = self.sentiment_analyzer.get_sentiment_summary(reviews)
                    doc['sentiment_summary'] = sentiment
            
            # Assess review quality if reviews are present
            if 'reviews' in doc and self.quality_assessor:
                reviews = doc['reviews']
                if reviews and len(reviews) > 0:
                    quality_scores = [
                        self.quality_assessor.assess_review(
                            r.get('text', ''), 
                            doc.get('title', '')
                        ) for r in reviews
                    ]
                    # Calculate average quality score
                    if quality_scores:
                        doc['avg_quality_score'] = np.mean([
                            s.authenticity_score * s.relevance_score 
                            for s in quality_scores
                        ])
            
            doc['processed'] = True
            processed.append(doc)
        
        return processed
    
    def _apply_personalization(self, 
                             docs: List[Dict], 
                             user_history: List[Dict]) -> List[Dict]:
        """Apply personalization based on user history."""
        # Extract user preferences from history
        viewed_products = set()
        purchased_products = set()
        wishlisted_products = set()
        
        for interaction in user_history:
            pid = interaction.get('product_id')
            if not pid:
                continue
                
            if interaction.get('type') == 'view':
                viewed_products.add(pid)
            elif interaction.get('type') == 'purchase':
                purchased_products.add(pid)
            elif interaction.get('type') == 'wishlist':
                wishlisted_products.add(pid)
        
        # Apply personalization scores
        for doc in docs:
            doc['personalization_score'] = 0.0
            pid = doc.get('product_id')
            
            if not pid:
                continue
                
            if pid in purchased_products:
                doc['personalization_score'] += 0.5
            if pid in wishlisted_products:
                doc['personalization_score'] += 0.3
            if pid in viewed_products:
                doc['personalization_score'] += 0.2
        
        return docs
    
    def _select_top_results(self, 
                          docs: List[Dict], 
                          k: int) -> List[Dict]:
        """Select top k results based on relevance and other factors."""
        if not docs:
            return []
        
        # Calculate composite score for each document
        for doc in docs:
            # Base score from vector similarity (if available)
            similarity_score = doc.get('similarity_score', 0.5)
            
            # Quality score (if available)
            quality_score = doc.get('avg_quality_score', 0.7)
            
            # Personalization score (if available)
            personalization_score = doc.get('personalization_score', 0.0)
            
            # Calculate composite score with weights
            doc['composite_score'] = (
                0.6 * similarity_score +
                0.2 * quality_score +
                0.2 * personalization_score
            )
        
        # Sort by composite score and select top k
        sorted_docs = sorted(
            docs, 
            key=lambda x: x.get('composite_score', 0), 
            reverse=True
        )
        
        return sorted_docs[:k]
    
    def _generate_response(self, 
                         query: str, 
                         docs: List[Dict]) -> str:
        """Generate a natural language response based on retrieved documents."""
        if not docs:
            return "I couldn't find any products matching your query. Could you provide more details?"
        
        # Extract key information from top documents
        product_info = []
        for doc in docs[:3]:  # Use top 3 for response
            info = {
                'name': doc.get('title', 'Unknown Product'),
                'brand': doc.get('brand', 'Unknown Brand'),
                'price': doc.get('price', 'Price not available'),
                'rating': doc.get('average_rating', 'No ratings yet')
            }
            
            # Add sentiment summary if available
            sentiment = doc.get('sentiment_summary', {})
            if sentiment:
                pos = sentiment.get('sentiment_distribution', {}).get('positive', 0)
                total = sum(sentiment.get('sentiment_distribution', {}).values())
                if total > 0:
                    info['positive_reviews'] = f"{pos/total*100:.0f}% positive"
            
            product_info.append(info)
        
        # Generate a simple response (in a real system, this would use an LLM)
        response_parts = ["Here are some recommendations based on your query:"]
        
        for i, info in enumerate(product_info, 1):
            response_parts.append(
                f"{i}. **{info['name']}** by {info['brand']} - ${info['price']} "
                f"(Rating: {info['rating']})"
            )
            
            if 'positive_reviews' in info:
                response_parts.append(f"   - {info['positive_reviews']} of customers loved it!")
        
        response_parts.append("\nWould you like more details about any of these products?")
        
        return "\n".join(response_parts)
