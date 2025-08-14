import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))

from src.data_processing.data_loader import ProductDataLoader
from src.embedding.embedding_generator import EmbeddingGenerator
from src.retrieval.vector_store import VectorStore
from src.generation.recommendation_engine import RecommendationEngine

def test_data_loading():
    print("Testing data loading...")
    data_loader = ProductDataLoader("data")
    products = data_loader.load_sample_data()
    print(f"Loaded {len(products)} sample products")
    for i, product in enumerate(products[:2], 1):  # Show first 2 products
        print(f"\nProduct {i}:")
        print(f"  Title: {product.title}")
        print(f"  Description: {product.description[:100]}...")
        print(f"  Price: ${product.price}")
        print(f"  Category: {product.category}")
        print(f"  Brand: {product.brand}")
        print(f"  Specifications: {list(product.specifications.items())[:2]}...")
        print(f"  Reviews: {[r['text'][:50] + '...' for r in product.reviews]}")
    return products

def test_embeddings(products):
    print("\nTesting embedding generation...")
    embedding_generator = EmbeddingGenerator()
    product_dicts = [p.to_dict() for p in products]
    embeddings = embedding_generator.get_product_embeddings(product_dicts)
    
    print(f"Generated embeddings for {len(embeddings)} products")
    product_id = next(iter(embeddings))
    print(f"\nSample embeddings for product '{products[0].title}':")
    for aspect, embedding in embeddings[product_id].items():
        print(f"  {aspect}: {embedding.shape} - {embedding[:3]}...")
    
    return embeddings

def test_vector_store(products, embeddings):
    print("\nTesting vector store...")
    vector_store = VectorStore("./chroma_db_test")
    product_dicts = [p.to_dict() for p in products]
    
    print("Adding products to vector store...")
    vector_store.add_products(product_dicts, embeddings)
    
    # Test search
    query = "wireless headphones with noise cancellation"
    print(f"\nSearching for: '{query}'")
    results = vector_store.search(query, n_results=2)
    
    print("\nSearch results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['metadata']['title']} (Score: {result['score']:.3f})")
        print(f"   {result['document'][:100]}...")
    
    return vector_store

def test_recommendation_engine(vector_store, products):
    print("\nTesting recommendation engine...")
    engine = RecommendationEngine(vector_store)
    
    # Test similar products
    product_id = products[0].product_id
    print(f"\nFinding products similar to: {products[0].title}")
    similar = engine.get_similar_products(product_id, n_results=2)
    
    print("\nSimilar products:")
    for i, rec in enumerate(similar, 1):
        print(f"{i}. {rec.title} (Score: {rec.score:.3f})")
        print(f"   {rec.explanation}")
    
    # Test search
    query = "affordable wireless earbuds"
    print(f"\nSearching for: '{query}'")
    results = engine.search_products(query, n_results=2)
    
    print("\nSearch results:")
    for i, rec in enumerate(results, 1):
        print(f"{i}. {rec.title} (Score: {rec.score:.3f})")
        print(f"   {rec.explanation}")

if __name__ == "__main__":
    print("=== Starting Component Tests ===\n")
    
    # Test data loading
    products = test_data_loading()
    
    # Test embedding generation
    embeddings = test_embeddings(products)
    
    # Test vector store
    vector_store = test_vector_store(products, embeddings)
    
    # Test recommendation engine
    test_recommendation_engine(vector_store, products)
    
    print("\n=== All tests completed successfully ===")
