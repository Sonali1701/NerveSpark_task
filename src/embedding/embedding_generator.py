from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

class EmbeddingGenerator:
    """Class to generate embeddings for product data using sentence transformers."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: Optional[str] = None):
        """Initialize the embedding generator with a pre-trained model.
        
        Args:
            model_name: Name of the sentence transformer model to use
            device: Device to run the model on ('cuda', 'mps', or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array containing the text embedding
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input texts to embed
            batch_size: Batch size for processing
            
        Returns:
            2D numpy array where each row is the embedding of the corresponding text
        """
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    
    def get_product_embeddings(self, products: List[Dict[str, Any]]) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate embeddings for multiple product fields.
        
        Args:
            products: List of product dictionaries
            
        Returns:
            Dictionary mapping product IDs to their field embeddings
        """
        product_embeddings = {}
        
        # Process each product
        for product in tqdm(products, desc="Generating product embeddings"):
            product_id = product['product_id']
            
            # Create text representations of different aspects
            title_text = product['title']
            description_text = product['description']
            
            # Create a combined text with specifications
            specs_text = ", ".join(f"{k}: {v}" for k, v in product['specifications'].items())
            
            # Combine all review texts
            reviews_text = " ".join(review['text'] for review in product['reviews'])
            
            # Generate embeddings for each aspect
            embeddings = {
                'title': self.get_embedding(title_text),
                'description': self.get_embedding(description_text),
                'specifications': self.get_embedding(specs_text),
                'reviews': self.get_embedding(reviews_text) if reviews_text else np.zeros(384)  # Default size for all-MiniLM-L6-v2
            }
            
            product_embeddings[product_id] = embeddings
            
        return product_embeddings
    
    def combine_embeddings(
        self, 
        embeddings: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Combine multiple embeddings with optional weighting.
        
        Args:
            embeddings: Dictionary of field name to embedding array
            weights: Optional dictionary of field names to weights (must sum to 1.0)
            
        Returns:
            Combined embedding as a numpy array
        """
        if weights is None:
            # Default equal weighting
            weights = {field: 1.0/len(embeddings) for field in embeddings}
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise ValueError("Sum of weights must be greater than 0")
            
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Weighted sum of embeddings
        combined = np.zeros_like(next(iter(embeddings.values())))  # Initialize with correct shape
        for field, embedding in embeddings.items():
            if field in weights:
                combined += weights[field] * embedding
                
        return combined
