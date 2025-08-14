import streamlit as st
import json
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.data_loader import ProductDataLoader
from src.embedding.embedding_generator import EmbeddingGenerator
from src.retrieval.vector_store import VectorStore
from src.generation.recommendation_engine import RecommendationEngine
from src.generation.product_comparison import ProductComparator
from src.generation.rag_pipeline import RAGPipeline, RAGResponse
from src.learning.preference_learner import PreferenceLearner
from src.analysis.sentiment_analyzer import SentimentAnalyzer, ReviewSentiment
from src.analysis.review_quality import ReviewQualityAssessor, ReviewQualityScore
from src.retrieval.vector_store import VectorStore
from src.embedding.embedding_generator import EmbeddingGenerator

# Set page config
st.set_page_config(
    page_title="E-commerce Product Recommendation",
    page_icon="🛍️",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.vector_store = None
    st.session_state.recommendation_engine = None
    st.session_state.products = {}
    st.session_state.user_history = []

def initialize_system():
    """Initialize the recommendation system components."""
    try:
        with st.spinner("Initializing recommendation system..."):
            # Initialize data loader and load sample data
            data_loader = ProductDataLoader()
            products = data_loader.load_sample_data()
            
            # Create a dictionary of Product objects for easy access
            products_dict = {p.product_id: p for p in products}
            
            # Initialize embedding generator
            embedding_generator = EmbeddingGenerator()
            
            # Initialize vector store
            vector_store = VectorStore()
            
            # Prepare product embeddings
            product_embeddings = {}
            products_list = []  # List version for the vector store
            
            for product in products:
                product_id = product.product_id
                
                # Create product dictionary for the vector store
                product_dict = {
                    'product_id': product_id,
                    'title': product.title,
                    'description': product.description,
                    'price': product.price,
                    'category': product.category,
                    'brand': product.brand,
                    'specifications': product.specifications,
                    'reviews': product.reviews
                }
                products_list.append(product_dict)
                
                # Prepare embeddings
                product_embeddings[product_id] = {
                    'title': embedding_generator.get_embedding(product.title),
                    'description': embedding_generator.get_embedding(product.description),
                    'specifications': embedding_generator.get_embedding(
                        ', '.join(f"{k}: {v}" for k, v in product.specifications.items())
                    ),
                    'reviews': embedding_generator.get_embedding(
                        ' '.join(r.get('text', '') for r in product.reviews) if product.reviews else ""
                    )
                }
            
            # Add products to the vector store
            vector_store.add_products(products_list, product_embeddings)
            
            # Initialize recommendation engine with just the vector store
            recommendation_engine = RecommendationEngine(vector_store)
            
            # Initialize preference learner
            preference_learner = PreferenceLearner('data/user_preferences')
            
            # Initialize analyzers (lazy load when needed)
            sentiment_analyzer = SentimentAnalyzer()
            review_quality_assessor = ReviewQualityAssessor()
            
            # Initialize RAG pipeline
            rag_pipeline = RAGPipeline(
                vector_store=vector_store,
                embedding_generator=embedding_generator,
                sentiment_analyzer=sentiment_analyzer,
                quality_assessor=review_quality_assessor
            )
            
            # Store in session state
            st.session_state.products = products_dict  # This is the dict of Product objects
            st.session_state.recommendation_engine = recommendation_engine
            st.session_state.preference_learner = preference_learner
            st.session_state.sentiment_analyzer = sentiment_analyzer
            st.session_state.review_quality_assessor = review_quality_assessor
            st.session_state.rag_pipeline = rag_pipeline
            st.session_state.initialized = True
            
            # Initialize user history if not exists
            if 'user_history' not in st.session_state:
                st.session_state.user_history = []
                
            # Initialize user ID (in a real app, this would come from authentication)
            if 'user_id' not in st.session_state:
                st.session_state.user_id = 'demo_user'  # Default user for demo
                
            # Track viewed products for this session
            if 'viewed_products' not in st.session_state:
                st.session_state.viewed_products = set()
                
            # Cache for sentiment analysis results
            if 'sentiment_cache' not in st.session_state:
                st.session_state.sentiment_cache = {}
                
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        st.session_state.initialized = False
        st.stop()

def display_product_card(product_id: str, score: float = None, explanation: str = None):
    """Display a product card with detailed information and RAG context."""
    if product_id not in st.session_state.products:
        st.error(f"Product {product_id} not found")
        return
        
    product = st.session_state.products[product_id]
    
    # Track product view in user history
    if hasattr(st.session_state, 'user_history'):
        st.session_state.user_history.append({
            'type': 'view',
            'product_id': product_id,
            'timestamp': str(datetime.now())
        })
    
    with st.container():
        # Create columns for the product card
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Display product image (placeholder)
            st.image(
                product.image_url if hasattr(product, 'image_url') else "https://via.placeholder.com/150",
                width=150
            )
            
            # Display score if provided
            if score is not None:
                # Map score to a color gradient from red to green
                color = f"hsl({int(score * 120)}, 75%, 50%)"  # 0-120 in HSL (red to green)
                st.markdown(
                    f"<div style='text-align: center;'>"
                    f"<span style='font-size: 0.8em; color: #666;'>Relevance</span><br>"
                    f"<span style='font-size: 1.2em; font-weight: bold; color: {color};'>{score*100:.0f}%</span>"
                    "</div>",
                    unsafe_allow_html=True
                )
        
        with col2:
            # Product title and basic info with RAG context
            st.subheader(product.title)
            st.caption(f"{product.brand} • {product.category}")
            
            # Price and rating with enhanced display
            col_price, col_rating, col_actions = st.columns([1, 1, 2])
            with col_price:
                st.metric("Price", f"${product.price:.2f}")
            with col_rating:
                if hasattr(product, 'average_rating') and product.average_rating > 0:
                    st.metric("Rating", f"{product.average_rating:.1f} ⭐")
            with col_actions:
                # Action buttons
                if st.button("🛒 Add to Cart", key=f"cart_{product_id}", use_container_width=True):
                    # Add to cart logic here
                    st.session_state.user_history.append({
                        'type': 'cart',
                        'product_id': product_id,
                        'timestamp': str(datetime.now())
                    })
                    st.rerun()
            
            # Enhanced product description with expandable details
            with st.expander("📝 Description", expanded=False):
                st.write(product.description)
                
                # Show RAG explanation if available
                if explanation:
                    st.info(f"💡 {explanation}")
            
            # Quick action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("❤️ Save", key=f"save_{product_id}", use_container_width=True):
                    # Save to wishlist logic
                    st.session_state.user_history.append({
                        'type': 'wishlist',
                        'product_id': product_id,
                        'timestamp': str(datetime.now())
                    })
                    st.rerun()
            with col2:
                if st.button("🔍 Compare", key=f"compare_{product_id}", use_container_width=True):
                    # Add to comparison logic
                    if 'comparison_products' not in st.session_state:
                        st.session_state.comparison_products = set()
                    st.session_state.comparison_products.add(product_id)
                    st.rerun()
            with col3:
                if st.button("📊 Details", key=f"details_{product_id}", use_container_width=True):
                    # Show more details
                    st.session_state.selected_product = product_id
                    st.rerun()

def show_product_comparison(product_ids):
    """Display a comparison of selected products."""
    if not product_ids:
        st.warning("Please select at least two products to compare.")
        return
    
    st.subheader("🔄 Product Comparison")
    
    # Get product details
    products_to_compare = [
        {**st.session_state.products[pid].to_dict(), 'product_id': pid}
        for pid in product_ids
    ]
    
    # Initialize comparator
    comparator = ProductComparator(products_to_compare)
    
    # Get comparison table
    comparison_table = comparator.get_comparison_table()
    
    # Display comparison table
    if comparison_table:
        # Convert to pandas DataFrame for better display
        import pandas as pd
        
        # Create a list of dictionaries for the table
        table_data = []
        for row in comparison_table:
            table_data.append({
                'Feature': row['feature'],
                **{st.session_state.products[pid].title: row[pid] for pid in product_ids}
            })
        
        # Display as a table
        st.dataframe(
            pd.DataFrame(table_data).set_index('Feature'),
            use_container_width=True,
            height=min(800, 50 + len(comparison_table) * 30)
        )
        
        # Show recommendation based on comparison
        st.subheader("🤖 Recommendation")
        
        # Simple recommendation based on comparison
        scores = comparator.get_recommendation()
        if scores:
            best_product_id = max(scores.items(), key=lambda x: x[1])[0]
            best_product = st.session_state.products[best_product_id]
            
            st.success(f"Based on the comparison, we recommend: **{best_product.title}**")
            
            # Show score breakdown
            with st.expander("See score breakdown"):
                st.write("Scores (higher is better):")
                for pid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"- {st.session_state.products[pid].title}: {score:.2f}")
        
        # Add option to customize feature weights
        with st.expander("🛠️ Customize Feature Weights"):
            st.write("Adjust the importance of each feature for the recommendation:")
            
            features = comparator.extract_features()
            weights = {}
            
            for feature in features:
                col1, col2 = st.columns([2, 3])
                with col1:
                    st.write(f"**{feature.name}**")
                    if feature.is_numeric:
                        st.caption(f"Higher is {'better' if feature.higher_is_better else 'worse'}")
                with col2:
                    weight = st.slider(
                        f"Weight for {feature.name}",
                        min_value=-1.0,
                        max_value=1.0,
                        value=0.5 if feature.higher_is_better else -0.5,
                        step=0.1,
                        key=f"weight_{feature.name}",
                        label_visibility="collapsed"
                    )
                    weights[feature.name] = weight
            
            if st.button("Update Recommendation"):
                custom_scores = comparator.get_recommendation(weights)
                if custom_scores:
                    best_custom_id = max(custom_scores.items(), key=lambda x: x[1])[0]
                    best_custom = st.session_state.products[best_custom_id]
                    st.success(f"Based on your preferences, we recommend: **{best_custom.title}**")
                    
                    st.write("Updated scores:")
                    for pid, score in sorted(custom_scores.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"- {st.session_state.products[pid].title}: {score:.2f}")
    else:
        st.warning("No comparable features found for the selected products.")

def main():
    """Main application function."""
    st.title("🛍️ E-commerce Product Recommendation")
    
    # Initialize the system if not already done
    if not st.session_state.get('initialized', False):
        initialize_system()
    
    # Ensure recommendation engine is available
    if not hasattr(st.session_state, 'recommendation_engine') or st.session_state.recommendation_engine is None:
        st.error("Failed to initialize recommendation engine. Please try refreshing the page.")
        st.stop()
        
    # Ensure products are loaded
    if not hasattr(st.session_state, 'products') or not st.session_state.products:
        st.error("No products found. Please check your data source.")
        st.stop()
    
    # Initialize selected products for comparison in session state
    if 'compare_products' not in st.session_state:
        st.session_state.compare_products = set()
    
    # Add to comparison button in sidebar
    with st.sidebar:
        st.divider()
        st.subheader("🔍 Product Comparison")
        
        # Show selected products for comparison
        if st.session_state.compare_products:
            st.write("Selected for comparison:")
            for pid in list(st.session_state.compare_products):
                product = st.session_state.products[pid]
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.image("https://via.placeholder.com/50?text=Product", width=30)
                with col2:
                    st.write(product.title)
            
            # Add button to view comparison
            if st.button("Compare Selected Products"):
                st.session_state.view = "comparison"
            
            # Add button to clear selection
            if st.button("Clear Selection"):
                st.session_state.compare_products = set()
                st.rerun()
        else:
            st.info("Select products to compare from the main view.")
    
    # Handle view state
    view = st.session_state.get('view', 'main')
    
    # Show back button if in comparison view
    if view == "comparison":
        if st.button("← Back to Products"):
            st.session_state.view = "main"
            st.rerun()
        
        # Show comparison view
        show_product_comparison(list(st.session_state.compare_products))
        return
    
    # User preference section in sidebar
    with st.sidebar.expander("🎯 Your Preferences", expanded=False):
        if hasattr(st.session_state, 'preference_learner'):
            # Get user's preference weights
            product_features = {
                pid: {
                    **{'price': p.price, 'category': p.category, 'brand': p.brand},
                    **p.specifications
                }
                for pid, p in st.session_state.products.items()
            }
            
            # Get user's preference weights
            preference_weights = st.session_state.preference_learner.get_user_preference_weights(
                st.session_state.user_id,
                product_features
            )
            
            # Display top preferences
            if preference_weights:
                st.write("Your current preferences:")
                top_prefs = sorted(
                    preference_weights.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:5]  # Show top 5
                
                for feature, weight in top_prefs:
                    if abs(weight) > 0.1:  # Only show significant preferences
                        emoji = "⬆️" if weight > 0 else "⬇️"
                        st.write(f"{emoji} {feature}: {abs(weight):.1f}")
            else:
                st.info("Interact with products to build your preferences!")
    
    # Search functionality with RAG
    st.subheader("Search for Products")
    search_query = st.text_input("What are you looking for?", key="search_query")
    
    if search_query:
        # Get recommendations using RAG pipeline
        with st.spinner("Analyzing your request and finding the best matches..."):
            # Get user history for personalization
            user_history = st.session_state.user_history if hasattr(st.session_state, 'user_history') else []
            
            # Get RAG response
            rag_response = st.session_state.rag_pipeline.generate_recommendation(
                query=search_query,
                user_history=user_history,
                num_results=5
            )
            
            # Display RAG response
            st.markdown("### Recommendations for You")
            st.markdown(rag_response.response_text)
            
            # Show product cards for recommended products
            st.subheader("Recommended Products")
            for doc in rag_response.source_documents:
                product_id = doc.get('product_id')
                if product_id and product_id in st.session_state.products:
                    display_product_card(
                        product_id,
                        score=doc.get('composite_score', 0.8)
                    )
            
            # Show debug info in expander
            with st.expander("ℹ️ How these recommendations were generated"):
                st.write("""
                These recommendations were generated using a Retrieval-Augmented Generation (RAG) system that:
                
                1. **Understands your query** using advanced natural language processing
                2. **Retrieves relevant products** based on multiple factors:
                   - Semantic similarity to your search
                   - Product quality and popularity
                   - Your personal preferences and history
                   - Review sentiment and authenticity
                
                The system combines these signals to provide the most relevant and personalized recommendations.
                """)
                
                if st.checkbox("Show technical details"):
                    st.json({
                        "query": search_query,
                        "num_retrieved": rag_response.context.get('num_retrieved', 0),
                        "num_filtered": rag_response.context.get('num_filtered', 0),
                        "model": rag_response.metadata.get('model', 'unknown'),
                        "timestamp": rag_response.metadata.get('timestamp')
                    })
            st.subheader("Featured Products")
            featured_products = list(st.session_state.products.values())[:6]
            
            cols = st.columns(3)
            for i, product in enumerate(featured_products):
                with cols[i % 3]:
                    display_product_card(product.product_id)
    
    # Debug info (can be toggled in settings)
    if st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.subheader("Debug Information")
        st.sidebar.json({
            "user_history": st.session_state.user_history,
            "num_products_loaded": len(st.session_state.products)
        })

if __name__ == "__main__":
    main()
