import pytest
from unittest.mock import MagicMock, patch
from src.generation.rag_pipeline import RAGPipeline, RAGResponse
from src.retrieval.vector_store import VectorStore
from src.embedding.embedding_generator import EmbeddingGenerator
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.analysis.review_quality import ReviewQualityAssessor, ReviewQualityScore
from src.data_processing.data_loader import ProductDataLoader
import numpy as np

@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore."""
    with patch('src.retrieval.vector_store.VectorStore') as mock_class:
        mock_store = MagicMock(spec=VectorStore)
        mock_class.return_value = mock_store
        
        # Mock the similarity search to return some dummy results
        mock_store.similarity_search.return_value = [
            {
                'id': 'p1',
                'document': 'Sample product 1',
                'metadata': {'title': 'Product 1', 'category': 'Electronics', 'brand': 'BrandA', 'price': 999.99},
                'distance': 0.1
            },
            {
                'id': 'p2',
                'document': 'Sample product 2',
                'metadata': {'title': 'Product 2', 'category': 'Electronics', 'brand': 'BrandB', 'price': 799.99},
                'distance': 0.15
            }
        ]
        
        yield mock_store

@pytest.fixture
def mock_embedding_generator():
    """Create a mock EmbeddingGenerator."""
    with patch('src.embedding.embedding_generator.EmbeddingGenerator') as mock_class:
        mock_gen = MagicMock()
        mock_gen.get_embedding.return_value = np.random.rand(384).tolist()  # Mock embedding vector
        mock_class.return_value = mock_gen
        yield mock_gen

@pytest.fixture
def mock_sentiment_analyzer():
    """Create a mock SentimentAnalyzer."""
    with patch('src.analysis.sentiment_analyzer.SentimentAnalyzer') as mock_class:
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_review.return_value = MagicMock(
            sentiment='positive',
            confidence=0.9,
            aspects={'quality': {'sentiment': 'positive', 'score': 0.8}}
        )
        mock_class.return_value = mock_analyzer
        yield mock_analyzer

@pytest.fixture
def mock_quality_assessor():
    """Create a mock ReviewQualityAssessor."""
    with patch('src.analysis.review_quality.ReviewQualityAssessor') as mock_class:
        mock_assessor = MagicMock()
        mock_assessor.assess_review.return_value = ReviewQualityScore(
            authenticity_score=0.9,
            relevance_score=0.85,
            flags=[],
            explanation="High quality review"
        )
        mock_class.return_value = mock_assessor
        yield mock_assessor

@pytest.fixture
def setup_rag_pipeline(mock_vector_store, mock_embedding_generator, mock_sentiment_analyzer, mock_quality_assessor):
    """Set up a test RAG pipeline with mocked dependencies."""
    return RAGPipeline(
        vector_store=mock_vector_store,
        embedding_generator=mock_embedding_generator,
        sentiment_analyzer=mock_sentiment_analyzer,
        quality_assessor=mock_quality_assessor
    )

def test_rag_pipeline_initialization(setup_rag_pipeline, mock_vector_store, mock_embedding_generator):
    """Test that the RAG pipeline initializes correctly."""
    rag = setup_rag_pipeline
    assert rag is not None
    assert hasattr(rag, 'generate_recommendation')
    
    # Verify dependencies were properly injected
    assert rag.vector_store == mock_vector_store
    assert rag.embedding_generator == mock_embedding_generator

def test_generate_recommendation_basic(setup_rag_pipeline, mock_embedding_generator, mock_vector_store):
    """Test generating recommendations with a basic query."""
    rag = setup_rag_pipeline
    query = "Find me a high-quality laptop"
    
    # Setup mock return values
    mock_embedding = [0.1] * 384  # Mock embedding vector
    mock_embedding_generator.get_embedding.return_value = mock_embedding
    
    # Test with no user history
    response = rag.generate_recommendation(query)
    
    # Verify the response structure
    assert isinstance(response, RAGResponse)
    assert hasattr(response, 'response_text')
    assert hasattr(response, 'context')
    assert hasattr(response, 'source_documents')
    assert hasattr(response, 'metadata')
    
    # Verify the vector store was called with the expected arguments
    mock_vector_store.similarity_search.assert_called_once()
    
    # Should return the mock recommendations we set up
    assert len(response.source_documents) == 2
    assert response.source_documents[0]['id'] == 'p1'
    assert response.source_documents[1]['id'] == 'p2'

def test_personalized_recommendations(setup_rag_pipeline):
    """Test that recommendations are personalized based on user history."""
    rag = setup_rag_pipeline
    
    # Create some user history
    user_history = [
        {'type': 'view', 'product_id': 'p1', 'timestamp': '2023-01-01'},
        {'type': 'purchase', 'product_id': 'p2', 'timestamp': '2023-01-02'},
        {'type': 'wishlist', 'product_id': 'p3', 'timestamp': '2023-01-03'},
    ]
    
    response = rag.generate_recommendation(
        query="Recommend me something I might like",
        user_history=user_history
    )
    
    # Should return personalized recommendations
    assert len(response.source_documents) > 0
    
    # Check that personalization score is included
    for doc in response.source_documents:
        assert 'personalization_score' in doc

def test_empty_query(setup_rag_pipeline):
    """Test behavior with an empty query."""
    rag = setup_rag_pipeline
    
    with pytest.raises(ValueError):
        rag.generate_recommendation("")

def test_no_results_scenario(setup_rag_pipeline):
    """Test behavior when no results match the query."""
    rag = setup_rag_pipeline
    
    # Use a very specific query that likely won't match anything
    response = rag.generate_recommendation("xyz123nonexistentproduct")
    
    # Should still return a valid response object
    assert isinstance(response, RAGResponse)
    # But with no source documents
    assert len(response.source_documents) == 0

# Add more integration tests as needed
