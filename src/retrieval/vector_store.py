import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

class VectorStore:
    """Class to handle vector storage and retrieval using ChromaDB."""
    
    def __init__(self, persist_directory: str = "./chroma_db", model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the ChromaDB data
            model_name: Name of the sentence transformer model for embeddings
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collections for different aspects
        self.collections = {}
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        
        # Define collection names for different aspects
        self.aspects = ['title', 'description', 'specifications', 'reviews', 'combined']
        
        # Initialize collections
        for aspect in self.aspects:
            self.collections[aspect] = self.client.get_or_create_collection(
                name=f"products_{aspect}",
                embedding_function=self.embedding_function
            )
    
    def add_products(self, products: List[Dict[str, Any]], embeddings: Dict[str, Dict[str, np.ndarray]]):
        """Add products to the vector store.
        
        Args:
            products: List of product dictionaries
            embeddings: Dictionary of product IDs to their aspect embeddings
        """
        # Prepare data for each aspect
        aspect_data = {aspect: {"ids": [], "documents": [], "metadatas": []} 
                      for aspect in self.aspects}
        
        # Process each product
        for product in products:
            product_id = product["product_id"]
            
            # Add to combined collection (all text concatenated)
            combined_text = (
                f"Title: {product['title']}\n"
                f"Description: {product['description']}\n"
                f"Specifications: {', '.join(f'{k}: {v}' for k, v in product['specifications'].items())}\n"
                f"Reviews: {' '.join(r['text'] for r in product['reviews'])}"
            )
            
            aspect_data["combined"]["ids"].append(product_id)
            aspect_data["combined"]["documents"].append(combined_text)
            aspect_data["combined"]["metadatas"].append({
                "title": product["title"],
                "category": product["category"],
                "brand": product["brand"],
                "price": product["price"]
            })
            
            # Add to individual aspect collections
            for aspect in ['title', 'description', 'specifications', 'reviews']:
                if aspect in embeddings.get(product_id, {}):
                    aspect_data[aspect]["ids"].append(product_id)
                    
                    # Format the document based on aspect type
                    if aspect == 'specifications':
                        doc = ', '.join(f"{k}: {v}" for k, v in product[aspect].items())
                    elif aspect == 'reviews':
                        # Convert reviews to a readable string format
                        review_texts = []
                        for i, review in enumerate(product[aspect], 1):
                            rating = review.get('rating', 0) * '⭐' or 'No rating'
                            text = review.get('text', '').replace('\n', ' ').strip()
                            review_texts.append(f"Review {i}: {rating}\n{text}")
                        doc = '\n\n'.join(review_texts) or "No reviews"
                    else:
                        doc = str(product[aspect])
                    
                    aspect_data[aspect]["documents"].append(doc)
                    aspect_data[aspect]["metadatas"].append({
                        "title": product["title"],
                        "category": product["category"],
                        "brand": product["brand"],
                        "price": product["price"]
                    })
        
        # Add to collections
        for aspect, data in aspect_data.items():
            if data["ids"]:  # Only if there are items to add
                self.collections[aspect].add(
                    ids=data["ids"],
                    documents=data["documents"],
                    metadatas=data["metadatas"]
                )
    
    def search(
        self, 
        query: str, 
        aspect: str = "combined",
        n_results: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar products based on a query.
        
        Args:
            query: Search query text
            aspect: Which aspect to search on ('title', 'description', 'specifications', 'reviews', 'combined')
            n_results: Number of results to return
            filter_conditions: Optional filters to apply (e.g., {"category": "Electronics"})
            
        Returns:
            List of matching products with scores
        """
        if aspect not in self.aspects:
            raise ValueError(f"Invalid aspect: {aspect}. Must be one of {self.aspects}")
        
        # Convert filter conditions to Chroma format
        where = {}
        if filter_conditions:
            where = {"$and": [{"metadata": {k: v}} for k, v in filter_conditions.items()]}
        
        # Perform the search
        results = self.collections[aspect].query(
            query_texts=[query],
            n_results=min(n_results, 20),
            where=where or None
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'score': float(1.0 - results['distances'][0][i]),  # Convert to similarity score
                'metadata': results['metadatas'][0][i],
                'document': results['documents'][0][i]
            })
            
        return formatted_results
    
    def get_similar_products(
        self, 
        product_id: str, 
        aspect: str = "combined",
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Find products similar to a given product.
        
        Args:
            product_id: ID of the product to find similar items for
            aspect: Which aspect to compare on
            n_results: Number of similar products to return
            
        Returns:
            List of similar products with scores
        """
        if aspect not in self.aspects:
            raise ValueError(f"Invalid aspect: {aspect}. Must be one of {self.aspects}")
        
        # Get the product's embedding
        results = self.collections[aspect].get(
            ids=[product_id],
            include=["embeddings", "metadatas"]
        )
        
        if not results['ids']:
            return []
            
        # Find similar items
        similar = self.collections[aspect].query(
            query_embeddings=results['embeddings'][0],
            n_results=n_results + 1,  # +1 because the product itself will be in results
            where={"id": {"$ne": product_id}}  # Exclude the product itself
        )
        
        # Format results
        formatted_results = []
        for i in range(len(similar['ids'][0])):
            if similar['ids'][0][i] != product_id:  # Make sure we don't include the product itself
                formatted_results.append({
                    'id': similar['ids'][0][i],
                    'score': float(1.0 - similar['distances'][0][i]),  # Convert to similarity score
                    'metadata': similar['metadatas'][0][i],
                    'document': similar['documents'][0][i]
                })
                
        return formatted_results[:n_results]  # Ensure we don't return more than requested
