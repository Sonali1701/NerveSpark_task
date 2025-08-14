"""Configuration file for pytest."""
import pytest
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture(scope="session")
def sample_products():
    """Fixture providing sample product data for testing."""
    from src.data_processing.data_loader import ProductDataLoader
    data_loader = ProductDataLoader()
    return data_loader.load_sample_data()

@pytest.fixture
def sample_review():
    """Fixture providing a sample review for testing."""
    return {
        "text": "This product is amazing! The quality is excellent and it works perfectly.",
        "rating": 5,
        "user_id": "test_user_123",
        "date": "2023-01-01"
    }

@pytest.fixture
def mock_user_history():
    """Fixture providing mock user history for testing."""
    return [
        {"type": "view", "product_id": "p1", "timestamp": "2023-01-01T10:00:00"},
        {"type": "purchase", "product_id": "p2", "timestamp": "2023-01-02T15:30:00"},
        {"type": "wishlist", "product_id": "p3", "timestamp": "2023-01-03T09:15:00"},
    ]

# Add any additional fixtures here
