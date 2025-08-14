import os
import json
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Product:
    """Class to represent a product with its attributes."""
    product_id: str
    title: str
    description: str
    price: float
    category: str
    brand: str
    specifications: Dict[str, str]
    reviews: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Product object to dictionary."""
        return {
            "product_id": self.product_id,
            "title": self.title,
            "description": self.description,
            "price": self.price,
            "category": self.category,
            "brand": self.brand,
            "specifications": self.specifications,
            "reviews": self.reviews
        }

class ProductDataLoader:
    """Class to load and process product data from various sources."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the data loader with the directory containing product data.
        
        Args:
            data_dir: Directory containing product data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_sample_data(self) -> List[Product]:
        """Load sample product data for demonstration purposes.
        
        Returns:
            List of Product objects
        """
        # This is sample data - in a real application, this would load from files/API
        sample_products = [
            Product(
                product_id="p1",
                title="Wireless Bluetooth Headphones",
                description="High-quality wireless headphones with noise cancellation and 30-hour battery life.",
                price=129.99,
                category="Electronics",
                brand="SoundMasters",
                specifications={
                    "Connectivity": "Bluetooth 5.0",
                    "Battery Life": "30 hours",
                    "Noise Cancellation": "Active",
                    "Water Resistance": "IPX4",
                    "Weight": "250g"
                },
                reviews=[
                    {"user_id": "u1", "rating": 5, "text": "Great sound quality!"},
                    {"user_id": "u2", "rating": 4, "text": "Good battery life but a bit heavy."}
                ]
            ),
            # Add more sample products as needed
        ]
        
        return sample_products
    
    def save_products_to_json(self, products: List[Product], filename: str = "products.json"):
        """Save list of products to a JSON file.
        
        Args:
            products: List of Product objects to save
            filename: Name of the output JSON file
        """
        output_path = self.data_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([p.to_dict() for p in products], f, indent=2)
    
    def load_products_from_json(self, filename: str = "products.json") -> List[Product]:
        """Load products from a JSON file.
        
        Args:
            filename: Name of the JSON file to load from
            
        Returns:
            List of Product objects
        """
        file_path = self.data_dir / filename
        if not file_path.exists():
            return []
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return [
            Product(
                product_id=item['product_id'],
                title=item['title'],
                description=item['description'],
                price=item['price'],
                category=item['category'],
                brand=item['brand'],
                specifications=item['specifications'],
                reviews=item['reviews']
            )
            for item in data
        ]
