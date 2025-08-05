import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import json
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import umap.umap_ as umap
import psutil


# =============================================================================
# DATASETS MODULE
# =============================================================================

def get_preloaded_datasets():
    """Return dictionary of available pre-loaded datasets"""
    return {
        "Movie Plot Summaries": "A collection of movie plot summaries from various genres",
        "Book Descriptions": "Book descriptions and summaries from different categories",
        "Product Reviews": "Customer reviews for various products",
        "News Headlines": "News headlines from different topics",
        "Scientific Abstracts": "Research paper abstracts from various fields"
    }

def create_movie_dataset():
    """Create movie plot summaries dataset"""
    movies = [
        {
            "text": "A young wizard discovers his magical heritage on his 11th birthday and attends a school of witchcraft and wizardry where he uncovers the truth about his parents' death.",
            "category": "Fantasy",
            "title": "Harry Potter and the Philosopher's Stone",
            "year": 2001
        },
        {
            "text": "In a future where humanity has colonized other planets, a blade runner must hunt down and retire four replicants who have escaped to Earth.",
            "category": "Science Fiction",
            "title": "Blade Runner",
            "year": 1982
        },
        {
            "text": "A computer hacker learns that reality as he knows it is actually a simulated world controlled by machines, and he must choose between the comfortable lie and the harsh truth.",
            "category": "Science Fiction",
            "title": "The Matrix",
            "year": 1999
        },
        {
            "text": "A mafia family's patriarch transfers control of his empire to his reluctant son, exploring themes of power, family loyalty, and the American Dream.",
            "category": "Crime Drama",
            "title": "The Godfather",
            "year": 1972
        },
        {
            "text": "A group of friends embark on a quest to destroy a powerful ring and defeat the dark lord who created it, facing numerous challenges along the way.",
            "category": "Fantasy",
            "title": "The Lord of the Rings: The Fellowship of the Ring",
            "year": 2001
        },
        {
            "text": "In the depths of space, the crew of a commercial starship encounters a deadly alien organism that threatens their survival.",
            "category": "Horror/Science Fiction",
            "title": "Alien",
            "year": 1979
        },
        {
            "text": "A former soldier suffering from memory loss discovers he's part of a secret government program and must uncover his true identity while being hunted.",
            "category": "Action Thriller",
            "title": "The Bourne Identity",
            "year": 2002
        },
        {
            "text": "Two detectives investigate a series of murders connected to the seven deadly sins, leading them into a dark psychological game with a methodical killer.",
            "category": "Crime Thriller",
            "title": "Se7en",
            "year": 1995
        },
        {
            "text": "A love story between a poor artist and a wealthy woman aboard the ill-fated RMS Titanic during its maiden voyage.",
            "category": "Romance Drama",
            "title": "Titanic",
            "year": 1997
        },
        {
            "text": "A team of scientists must prevent a massive asteroid from colliding with Earth, racing against time to save humanity from extinction.",
            "category": "Action/Science Fiction",
            "title": "Armageddon",
            "year": 1998
        },
        {
            "text": "An insomniac office worker forms an underground fight club with a soap maker, leading to an anarchist organization that challenges consumer culture.",
            "category": "Drama Thriller",
            "title": "Fight Club",
            "year": 1999
        },
        {
            "text": "A brilliant mathematician struggles with schizophrenia while making groundbreaking discoveries in game theory and cryptography.",
            "category": "Biography Drama",
            "title": "A Beautiful Mind",
            "year": 2001
        },
        {
            "text": "In a post-apocalyptic wasteland, a former police officer helps a group of survivors escape from a tyrannical warlord in high-speed desert chases.",
            "category": "Action",
            "title": "Mad Max: Fury Road",
            "year": 2015
        },
        {
            "text": "A group of dreams-within-dreams specialists perform extraction and implantation of ideas in people's subconscious minds through shared dreaming.",
            "category": "Science Fiction Thriller",
            "title": "Inception",
            "year": 2010
        },
        {
            "text": "A jazz musician's obsession with perfection leads him to make a deal that transforms his life in unexpected ways while pursuing his dreams.",
            "category": "Musical Drama",
            "title": "Whiplash",
            "year": 2014
        }
    ]
    return pd.DataFrame(movies)

def create_book_dataset():
    """Create book descriptions dataset"""
    books = [
        {
            "text": "A dystopian novel about a society where books are banned and burned, following a fireman who begins to question his role in destroying literature.",
            "category": "Dystopian Fiction",
            "title": "Fahrenheit 451",
            "author": "Ray Bradbury"
        },
        {
            "text": "Set in the 1930s American South, this novel explores racial injustice through the eyes of a young girl whose father defends a Black man falsely accused of rape.",
            "category": "Historical Fiction",
            "title": "To Kill a Mockingbird",
            "author": "Harper Lee"
        },
        {
            "text": "A epic fantasy saga about the struggle for power in the Seven Kingdoms, featuring multiple storylines of noble families vying for the Iron Throne.",
            "category": "Epic Fantasy",
            "title": "A Game of Thrones",
            "author": "George R.R. Martin"
        },
        {
            "text": "A psychological thriller about an unreliable narrator who becomes obsessed with her ex-husband's new wife, leading to shocking revelations.",
            "category": "Psychological Thriller",
            "title": "Gone Girl",
            "author": "Gillian Flynn"
        },
        {
            "text": "A coming-of-age story about a young woman navigating family expectations, love, and personal identity in 19th century New England.",
            "category": "Classic Literature",
            "title": "Little Women",
            "author": "Louisa May Alcott"
        },
        {
            "text": "A science fiction novel exploring artificial intelligence, virtual reality, and the nature of consciousness in a near-future cyberpunk world.",
            "category": "Cyberpunk",
            "title": "Neuromancer",
            "author": "William Gibson"
        },
        {
            "text": "A historical novel about a midwife in colonial America who faces accusations of witchcraft while trying to help women in her community.",
            "category": "Historical Fiction",
            "title": "The Midwife's Apprentice",
            "author": "Karen Cushman"
        },
        {
            "text": "A mystery novel featuring a detective investigating murders on a remote island where guests disappear one by one following a sinister pattern.",
            "category": "Mystery",
            "title": "And Then There Were None",
            "author": "Agatha Christie"
        },
        {
            "text": "A space opera following a young farm boy who discovers his connection to an ancient mystical force and joins a rebellion against an evil empire.",
            "category": "Space Opera",
            "title": "Star Wars: A New Hope (novelization)",
            "author": "George Lucas"
        },
        {
            "text": "A romantic novel about a strong-willed Southern woman who struggles to survive during the American Civil War and Reconstruction era.",
            "category": "Historical Romance",
            "title": "Gone with the Wind",
            "author": "Margaret Mitchell"
        },
        {
            "text": "A philosophical novel about a young prince's journey to different planets, exploring themes of love, friendship, and the meaning of life.",
            "category": "Philosophical Fiction",
            "title": "The Little Prince",
            "author": "Antoine de Saint-Exupéry"
        },
        {
            "text": "A dark fantasy series about a gunslinger's quest to reach the Dark Tower, blending Western, fantasy, and horror elements across multiple worlds.",
            "category": "Dark Fantasy",
            "title": "The Dark Tower: The Gunslinger",
            "author": "Stephen King"
        }
    ]
    return pd.DataFrame(books)

def create_product_reviews_dataset():
    """Create product reviews dataset"""
    reviews = [
        {
            "text": "This smartphone has an amazing camera quality and the battery lasts all day. The interface is intuitive and responsive. Highly recommended for photography enthusiasts.",
            "category": "Electronics",
            "product": "Smartphone",
            "rating": 5
        },
        {
            "text": "The coffee maker brews excellent coffee but the water reservoir is quite small. Good build quality overall, though the price point is a bit high for the features offered.",
            "category": "Appliances",
            "product": "Coffee Maker",
            "rating": 4
        },
        {
            "text": "This laptop is perfect for gaming with its powerful graphics card and fast processor. However, it gets quite hot during intensive use and the fan can be noisy.",
            "category": "Electronics",
            "product": "Gaming Laptop",
            "rating": 4
        },
        {
            "text": "The wireless headphones have incredible sound quality and noise cancellation. The battery life is impressive, lasting over 20 hours on a single charge.",
            "category": "Electronics",
            "product": "Wireless Headphones",
            "rating": 5
        },
        {
            "text": "This vacuum cleaner is lightweight and easy to maneuver. It picks up pet hair very well, but the dust container needs frequent emptying on larger cleaning jobs.",
            "category": "Appliances",
            "product": "Vacuum Cleaner",
            "rating": 4
        },
        {
            "text": "The running shoes are comfortable and provide good support during long runs. The breathable material keeps feet dry, though they show wear faster than expected.",
            "category": "Sports",
            "product": "Running Shoes",
            "rating": 4
        },
        {
            "text": "This skincare product has made a noticeable difference in my skin texture and appearance. It's gentle on sensitive skin and doesn't cause breakouts.",
            "category": "Beauty",
            "product": "Face Serum",
            "rating": 5
        },
        {
            "text": "The kitchen knife set is sharp and well-balanced. The handles are ergonomic and comfortable to use. Great value for money with excellent build quality.",
            "category": "Kitchen",
            "product": "Knife Set",
            "rating": 5
        },
        {
            "text": "This fitness tracker accurately monitors heart rate and sleep patterns. The app interface is user-friendly, but the wristband material can cause irritation during extended wear.",
            "category": "Electronics",
            "product": "Fitness Tracker",
            "rating": 4
        },
        {
            "text": "The mattress provides excellent support and comfort. It has significantly improved my sleep quality, though it took a few weeks to adjust to the firmness level.",
            "category": "Home",
            "product": "Memory Foam Mattress",
            "rating": 5
        },
        {
            "text": "This board game is entertaining and engaging for the whole family. The rules are easy to learn and it offers good replay value with different strategies to explore.",
            "category": "Toys & Games",
            "product": "Strategy Board Game",
            "rating": 5
        },
        {
            "text": "The camping tent is spacious and weather-resistant. Setup is straightforward, but the included stakes are flimsy and should be replaced with sturdier alternatives.",
            "category": "Outdoor",
            "product": "Camping Tent",
            "rating": 4
        }
    ]
    return pd.DataFrame(reviews)

def create_news_dataset():
    """Create news headlines dataset"""
    news = [
        {
            "text": "Scientists discover new treatment method for Alzheimer's disease that shows promising results in clinical trials, offering hope for millions of patients worldwide.",
            "category": "Health & Science",
            "topic": "Medical Research"
        },
        {
            "text": "Major technology company announces breakthrough in quantum computing, achieving significant improvements in processing speed and error reduction.",
            "category": "Technology",
            "topic": "Computing"
        },
        {
            "text": "International climate summit reaches historic agreement on carbon emission reduction targets, with 195 countries committing to ambitious goals.",
            "category": "Environment",
            "topic": "Climate Change"
        },
        {
            "text": "Stock markets experience significant volatility as investors react to unexpected changes in interest rates and inflation concerns.",
            "category": "Business & Finance",
            "topic": "Markets"
        },
        {
            "text": "Archaeological team uncovers ancient civilization remains that could rewrite understanding of early human migration patterns and cultural development.",
            "category": "Science & History",
            "topic": "Archaeology"
        },
        {
            "text": "Professional sports league implements new safety protocols following comprehensive study on player health and injury prevention measures.",
            "category": "Sports",
            "topic": "Player Safety"
        },
        {
            "text": "Educational institutions adopt innovative teaching methods using virtual reality technology to enhance student learning experiences across various subjects.",
            "category": "Education",
            "topic": "Educational Technology"
        },
        {
            "text": "Renewable energy project reaches milestone with completion of world's largest solar power installation, capable of supplying electricity to millions of homes.",
            "category": "Environment",
            "topic": "Renewable Energy"
        },
        {
            "text": "Government announces new infrastructure investment plan focusing on transportation improvements, including high-speed rail and electric vehicle charging networks.",
            "category": "Politics",
            "topic": "Infrastructure"
        },
        {
            "text": "Space exploration mission successfully lands on distant planet, transmitting valuable scientific data about atmospheric conditions and potential for life.",
            "category": "Science & Technology",
            "topic": "Space Exploration"
        },
        {
            "text": "Global health organization reports significant progress in vaccination campaigns, leading to decreased infection rates and improved public health outcomes.",
            "category": "Health",
            "topic": "Public Health"
        },
        {
            "text": "Artificial intelligence system demonstrates remarkable capabilities in medical diagnosis, showing higher accuracy rates than traditional methods in detecting diseases.",
            "category": "Technology & Health",
            "topic": "AI in Medicine"
        }
    ]
    return pd.DataFrame(news)

def create_scientific_abstracts_dataset():
    """Create scientific abstracts dataset"""
    abstracts = [
        {
            "text": "This study investigates the application of machine learning algorithms in predicting protein folding patterns. Using deep neural networks trained on extensive protein databases, we achieved 94% accuracy in secondary structure prediction, significantly improving upon existing methods.",
            "category": "Computational Biology",
            "field": "Bioinformatics"
        },
        {
            "text": "We present a novel approach to quantum error correction using topological qubits that demonstrates enhanced stability against decoherence. Our experimental results show a 10-fold improvement in error rates compared to conventional superconducting qubit implementations.",
            "category": "Quantum Physics",
            "field": "Quantum Computing"
        },
        {
            "text": "This research explores the synthesis of advanced nanomaterials for energy storage applications. We developed a new class of graphene-based composites that exhibit superior capacitance and cycling stability for supercapacitor applications.",
            "category": "Materials Science",
            "field": "Nanotechnology"
        },
        {
            "text": "Our analysis of climate data from the past century reveals accelerating trends in global temperature increase and precipitation pattern changes. The study provides new insights into regional climate variability and extreme weather event frequency.",
            "category": "Environmental Science",
            "field": "Climate Research"
        },
        {
            "text": "We report the development of a breakthrough gene therapy technique for treating inherited retinal diseases. Clinical trials show significant vision improvement in 85% of patients with previously untreatable conditions.",
            "category": "Medical Research",
            "field": "Gene Therapy"
        },
        {
            "text": "This study examines the effectiveness of natural language processing models in cross-cultural communication analysis. Our multilingual transformer architecture achieves state-of-the-art performance in sentiment analysis across 12 different languages.",
            "category": "Computer Science",
            "field": "Natural Language Processing"
        },
        {
            "text": "We investigate the neurological mechanisms underlying memory formation and consolidation using advanced brain imaging techniques. Our findings reveal new pathways involved in long-term memory storage and retrieval processes.",
            "category": "Neuroscience",
            "field": "Cognitive Neuroscience"
        },
        {
            "text": "This research presents a comprehensive analysis of renewable energy integration challenges in modern power grids. We propose adaptive control strategies that optimize energy distribution while maintaining grid stability and efficiency.",
            "category": "Engineering",
            "field": "Power Systems"
        },
        {
            "text": "Our study explores the application of CRISPR-Cas9 technology in developing disease-resistant crop varieties. Field trials demonstrate significant improvements in yield and pathogen resistance without compromising nutritional quality.",
            "category": "Agricultural Science",
            "field": "Plant Biotechnology"
        },
        {
            "text": "We present evidence for the existence of previously unknown exoplanets in nearby star systems using improved detection methods. Spectroscopic analysis suggests potential habitability conditions on three newly discovered worlds.",
            "category": "Astronomy",
            "field": "Exoplanet Research"
        },
        {
            "text": "This investigation focuses on the development of biodegradable plastics from marine waste materials. Our innovative processing technique produces polymers with comparable strength to conventional plastics while addressing ocean pollution.",
            "category": "Environmental Engineering",
            "field": "Sustainable Materials"
        },
        {
            "text": "We examine the psychological factors influencing decision-making in high-stress environments through controlled laboratory experiments. Results indicate that specific cognitive training methods can significantly improve performance under pressure.",
            "category": "Psychology",
            "field": "Cognitive Psychology"
        }
    ]
    return pd.DataFrame(abstracts)

def load_dataset(dataset_name):
    """Load a specific pre-loaded dataset"""
    if dataset_name == "Movie Plot Summaries":
        return create_movie_dataset()
    elif dataset_name == "Book Descriptions":
        return create_book_dataset()
    elif dataset_name == "Product Reviews":
        return create_product_reviews_dataset()
    elif dataset_name == "News Headlines":
        return create_news_dataset()
    elif dataset_name == "Scientific Abstracts":
        return create_scientific_abstracts_dataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# =============================================================================
# EMBEDDING GENERATOR CLASS
# =============================================================================

class EmbeddingGenerator:
    """Handle embedding generation using gte-modernbert-base model"""

    def __init__(self):
        self.model = None
        self.model_name = "Alibaba-NLP/gte-modernbert-base"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Load the embedding model"""
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                return f"Model loaded successfully on {self.device}"
            except Exception as e:
                # Fallback to a smaller model if gte-modernbert-base fails
                try:
                    self.model_name = "all-MiniLM-L6-v2"
                    self.model = SentenceTransformer(self.model_name, device=self.device)
                    return f"Fallback to {self.model_name}"
                except Exception as fallback_error:
                    return f"Both models failed: {str(fallback_error)}"

    def generate_embeddings(self, texts, batch_size=32, progress_callback=None):
        """Generate embeddings for a list of texts"""
        if self.model is None:
            self.load_model()

        try:
            # Process in batches with progress tracking
            embeddings = []
            total_batches = len(texts) // batch_size + (1 if len(texts) % batch_size != 0 else 0)

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]

                # Generate embeddings for batch
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )

                embeddings.append(batch_embeddings)

                # Update progress
                if progress_callback:
                    progress = ((i // batch_size) + 1) / total_batches * 100
                    progress_callback(progress)

            # Concatenate all embeddings
            final_embeddings = np.vstack(embeddings)
            return final_embeddings

        except Exception as e:
            raise e

    def encode_single(self, text):
        """Encode a single text string"""
        if self.model is None:
            self.load_model()

        try:
            embedding = self.model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding[0]
        except Exception as e:
            raise e

# =============================================================================
# SIMILARITY SEARCHER CLASS
# =============================================================================

class SimilaritySearcher:
    """Handle similarity search operations"""

    def __init__(self):
        self.embeddings = None
        self.texts = None

    def find_similar(self, query_embedding, embeddings, threshold=0.3, top_k=10):
        """Find similar texts based on cosine similarity"""
        try:
            # Ensure query_embedding is 2D for sklearn
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            # Calculate cosine similarities
            similarities = cosine_similarity(query_embedding, embeddings)[0]

            # Filter by threshold
            valid_indices = np.where(similarities >= threshold)[0]
            valid_similarities = similarities[valid_indices]

            # Sort by similarity (descending)
            sorted_indices = np.argsort(valid_similarities)[::-1]

            # Limit to top_k
            if len(sorted_indices) > top_k:
                sorted_indices = sorted_indices[:top_k]

            final_indices = valid_indices[sorted_indices]
            final_similarities = valid_similarities[sorted_indices]

            return final_indices, final_similarities

        except Exception as e:
            return np.array([]), np.array([])

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_umap_visualization(embeddings, data_df=None, umap_embeddings=None,
                              color_by='category', point_size=8, opacity=0.7):
    """Create UMAP visualization of embeddings"""
    try:
        # Compute UMAP if not provided
        if umap_embeddings is None:
            reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                n_components=2,
                random_state=42,
                metric='cosine'
            )
            umap_embeddings = reducer.fit_transform(embeddings)

        # If only embeddings requested, return them
        if data_df is None:
            return umap_embeddings

        # Create visualization DataFrame
        viz_df = pd.DataFrame({
            'x': umap_embeddings[:, 0],
            'y': umap_embeddings[:, 1],
            'text': data_df['text'].str[:100] + '...',  # Truncate for hover
            'full_text': data_df['text'],
            'index': range(len(data_df))
        })

        # Add metadata columns if they exist
        for col in data_df.columns:
            if col != 'text':
                viz_df[col] = data_df[col]

        # Handle color mapping
        if color_by == 'none':
            color_column = None
        elif color_by == 'text_length':
            viz_df['text_length'] = data_df['text'].str.len()
            color_column = 'text_length'
        elif color_by in viz_df.columns:
            color_column = color_by
        else:
            color_column = None

        # Create the plot
        if color_column and color_column == 'text_length':
            # Continuous color scale for text length
            fig = px.scatter(
                viz_df,
                x='x',
                y='y',
                color='text_length',
                hover_data=['text'],
                color_continuous_scale='Viridis',
                opacity=opacity,
                title='2D Embedding Visualization (UMAP)'
            )
        elif color_column:
            # Discrete color scale for categories
            fig = px.scatter(
                viz_df,
                x='x',
                y='y',
                color=color_column,
                hover_data=['text'],
                opacity=opacity,
                title='2D Embedding Visualization (UMAP)'
            )
        else:
            # No color coding
            fig = px.scatter(
                viz_df,
                x='x',
                y='y',
                hover_data=['text'],
                opacity=opacity,
                title='2D Embedding Visualization (UMAP)'
            )

        # Update traces
        fig.update_traces(marker=dict(size=point_size))

        # Update layout
        fig.update_layout(
            width=800,
            height=600,
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            showlegend=True if color_column and color_column != 'text_length' else False,
            hovermode='closest'
        )

        return fig

    except Exception as e:
        return None

# =============================================================================
# GLOBAL STATE VARIABLES
# =============================================================================

# Global variables to store state
current_dataset = None
embeddings = None
umap_embeddings = None
embedding_generator = EmbeddingGenerator()
similarity_searcher = SimilaritySearcher()

document_intelligence = None

# =============================================================================
# MAIN APPLICATION FUNCTIONS
# =============================================================================

def load_preloaded_dataset(dataset_name):
    """Load a preloaded dataset"""
    global current_dataset, embeddings, umap_embeddings

    try:
        current_dataset = load_dataset(dataset_name)
        embeddings = None
        umap_embeddings = None

        # Create summary
        summary = f"""
Dataset loaded successfully!

**Dataset:** {dataset_name}
**Total texts:** {len(current_dataset)}
**Columns:** {', '.join(current_dataset.columns)}

**Sample entries:**
"""
        for i, row in current_dataset.head(3).iterrows():
            summary += f"\n{i+1}. {row['text'][:100]}..."

        return summary, current_dataset

    except Exception as e:
        return f"Error loading dataset: {str(e)}", None

def load_custom_file(file):
    """Load custom CSV or JSON file"""
    global current_dataset, embeddings, umap_embeddings

    if file is None:
        return "No file uploaded", None

    try:
        if file.name.endswith('.csv'):
            current_dataset = pd.read_csv(file.name)
            if 'text' not in current_dataset.columns:
                return "Error: CSV must contain a 'text' column", None
        elif file.name.endswith('.json'):
            with open(file.name, 'r') as f:
                data = json.load(f)
            if isinstance(data, list) and all('text' in item for item in data):
                current_dataset = pd.DataFrame(data)
            else:
                return "Error: JSON must be array of objects with 'text' field", None
        else:
            return "Error: File must be CSV or JSON", None

        embeddings = None
        umap_embeddings = None

        summary = f"""
Custom file loaded successfully!

**Filename:** {file.name}
**Total texts:** {len(current_dataset)}
**Columns:** {', '.join(current_dataset.columns)}

**Sample entries:**
"""
        for i, row in current_dataset.head(3).iterrows():
            summary += f"\n{i+1}. {row['text'][:100]}..."

        return summary, current_dataset

    except Exception as e:
        return f"Error loading file: {str(e)}", None

def process_custom_texts(custom_texts):
    """Process custom text input"""
    global current_dataset, embeddings, umap_embeddings

    if not custom_texts or not custom_texts.strip():
        return "No texts provided", None

    try:
        texts = [text.strip() for text in custom_texts.split('\n') if text.strip()]
        if not texts:
            return "No valid texts found", None

        current_dataset = pd.DataFrame({
            'text': texts,
            'category': ['Custom'] * len(texts),
            'id': range(len(texts))
        })

        embeddings = None
        umap_embeddings = None

        summary = f"""
Custom texts processed successfully!

**Total texts:** {len(texts)}

**Entries:**
"""
        for i, text in enumerate(texts[:5]):
            summary += f"\n{i+1}. {text[:100]}..."

        if len(texts) > 5:
            summary += f"\n... and {len(texts) - 5} more"

        return summary, current_dataset

    except Exception as e:
        return f"Error processing texts: {str(e)}", None

def generate_embeddings_func():
    """Generate embeddings for current dataset"""
    global embeddings, umap_embeddings

    if current_dataset is None:
        return "No dataset loaded. Please load a dataset first."

    try:
        start_time = time.time()

        # Load model if needed
        model_status = embedding_generator.load_model()

        texts = current_dataset['text'].tolist()

        # Generate embeddings
        embeddings = embedding_generator.generate_embeddings(texts)

        # Generate UMAP embeddings for visualization
        umap_embeddings = create_umap_visualization(embeddings, data_df=None)

        generation_time = time.time() - start_time

        return f"""
Embeddings generated successfully!

**Model:** {embedding_generator.model_name}
**Device:** {embedding_generator.device}
**Dataset size:** {len(texts)}
**Embedding dimension:** {embeddings.shape[1]}
**Generation time:** {generation_time:.2f} seconds
**Processing speed:** {len(texts) / generation_time:.1f} texts/second
**Memory usage:** {embeddings.nbytes / (1024 * 1024):.1f} MB

{model_status}
"""

    except Exception as e:
        return f"Error generating embeddings: {str(e)}"

def search_similar_texts(query_text, similarity_threshold, max_results):
    """Search for similar texts"""
    global embeddings, current_dataset

    if embeddings is None:
        return "No embeddings generated. Please generate embeddings first.", None

    if not query_text or not query_text.strip():
        return "Please enter a search query.", None

    try:
        start_time = time.time()

        # Generate query embedding
        query_embedding = embedding_generator.encode_single(query_text)

        # Find similar texts
        similar_indices, similarities = similarity_searcher.find_similar(
            query_embedding,
            embeddings,
            threshold=similarity_threshold,
            top_k=max_results
        )

        search_time = time.time() - start_time

        if len(similar_indices) == 0:
            return f"No texts found with similarity > {similarity_threshold:.2f}", None

        # Create results dataframe
        results_df = current_dataset.iloc[similar_indices].copy()
        results_df['similarity_score'] = similarities
        results_df = results_df.sort_values('similarity_score', ascending=False)

        # Format for display
        display_columns = ['text', 'similarity_score']
        if 'category' in results_df.columns:
            display_columns.insert(-1, 'category')

        results_display = results_df[display_columns].round({'similarity_score': 3})

        summary = f"""
Search Results

**Query:** {query_text}
**Found:** {len(similar_indices)} similar texts
**Search time:** {search_time:.3f} seconds
**Similarity threshold:** {similarity_threshold:.2f}

**Top Results:**
"""

        for i, (_, row) in enumerate(results_display.head(5).iterrows()):
            summary += f"\n{i+1}. **Score: {row['similarity_score']:.3f}**"
            if 'category' in row:
                summary += f" | Category: {row['category']}"
            summary += f"\n   {row['text'][:150]}...\n"

        return summary, results_display

    except Exception as e:
        return f"Error during search: {str(e)}", None

def create_visualization(color_by, point_size, opacity):
    """Create UMAP visualization"""
    if embeddings is None or current_dataset is None:
        return None

    try:
        fig = create_umap_visualization(
            embeddings,
            current_dataset,
            umap_embeddings=umap_embeddings,
            color_by=color_by,
            point_size=point_size,
            opacity=opacity
        )
        return fig
    except Exception as e:
        return None

def export_data(include_embeddings, include_umap):
    """Export data to CSV"""
    if current_dataset is None:
        return None

    try:
        export_df = current_dataset.copy()

        if include_embeddings and embeddings is not None:
            # Add embeddings as columns
            for i in range(embeddings.shape[1]):
                export_df[f'emb_{i}'] = embeddings[:, i]

        if include_umap and umap_embeddings is not None:
            export_df['umap_x'] = umap_embeddings[:, 0]
            export_df['umap_y'] = umap_embeddings[:, 1]

        # Save to temporary file
        filename = f"embeddings_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        export_df.to_csv(filename, index=False)

        return filename

    except Exception as e:
        return None

def get_dataset_overview():
    """Get dataset overview and statistics"""
    if current_dataset is None:
        return "No dataset loaded"

    try:
        df = current_dataset

        overview = f"""
# Dataset Overview

**Total Texts:** {len(df)}
**Average Text Length:** {df['text'].str.len().mean():.0f} characters
**Columns:** {', '.join(df.columns)}
"""

        if 'category' in df.columns:
            overview += f"**Categories:** {df['category'].nunique()}\n\n"
            overview += "**Category Distribution:**\n"
            category_counts = df['category'].value_counts()
            for cat, count in category_counts.items():
                overview += f"- {cat}: {count}\n"

        overview += "\n**Sample Texts:**\n"
        for i, row in df.head(5).iterrows():
            overview += f"\n{i+1}. "
            if 'category' in row:
                overview += f"[{row['category']}] "
            overview += f"{row['text'][:200]}...\n"

        return overview

    except Exception as e:
        return f"Error generating overview: {str(e)}"

def get_performance_metrics():
    """Get performance metrics"""
    if embeddings is None:
        return "No embeddings generated yet"

    try:
        metrics = f"""
# Performance Metrics

**Dataset Size:** {len(current_dataset)} texts
**Embedding Dimension:** {embeddings.shape[1]}
**Model:** {embedding_generator.model_name}
**Device:** {embedding_generator.device}
**Memory Usage:** {embeddings.nbytes / (1024 * 1024):.1f} MB

**System Information:**
"""

        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            metrics += f"- **Process Memory:** {memory_info.rss / (1024 * 1024):.1f} MB\n"
            metrics += f"- **CPU Count:** {psutil.cpu_count()}\n"
            metrics += f"- **Available Memory:** {psutil.virtual_memory().available / (1024**3):.1f} GB\n"
        except:
            metrics += "- System info unavailable\n"

        return metrics

    except Exception as e:
        return f"Error getting metrics: {str(e)}"

def analyze_corpus_intelligence(documents_text):
    """Analyze document corpus with intelligence features"""
    global embedding_generator

    if not documents_text.strip():
        return "Please provide documents to analyze."

    # Split documents by double newline or numbered format
    if '\n\n' in documents_text:
        documents = [doc.strip() for doc in documents_text.split('\n\n') if doc.strip()]
    else:
        documents = [documents_text]  # Treat as single document

    # Create DocumentIntelligence instance
    from document_intelligence import DocumentIntelligence
    di = DocumentIntelligence(embedding_generator)

    try:
        analysis = di.analyze_document_corpus(documents)

        # Format results for display
        result = f"""
# Document Intelligence Analysis

## Corpus Statistics
- **Documents**: {analysis['corpus_stats']['document_count']}
- **Total Words**: {analysis['corpus_stats']['total_words']:,}
- **Vocabulary Size**: {analysis['corpus_stats']['vocabulary_size']:,}
- **Avg Words/Doc**: {analysis['corpus_stats']['average_words_per_document']:.1f}

## Key Phrases
"""

        for i, phrase in enumerate(analysis['key_phrases'][:10], 1):
            result += f"{i}. **{phrase['phrase']}** (score: {phrase['score']:.2f})\n"

        return result

    except Exception as e:
        return f"Error analyzing corpus: {str(e)}"

def summarize_document_interface(document_text, summary_length):
    """Summarize document interface"""
    if not document_text.strip():
        return "Please provide a document to summarize."

    from document_intelligence import DocumentIntelligence
    di = DocumentIntelligence()

    try:
        summary = di.summarize_document(document_text, summary_length)

        return f"""
# Document Summary

**Original Length**: {len(document_text)} characters
**Summary Length**: {len(summary['summary'])} characters
**Compression Ratio**: {summary['compression_ratio']:.2f}

## Summary:
{summary['summary']}
"""
    except Exception as e:
        return f"Error summarizing document: {str(e)}"

def answer_question_interface(question, context_text):
    """Question answering interface"""
    if not question.strip() or not context_text.strip():
        return "Please provide both a question and context documents."

    # Split context into documents
    if '\n\n' in context_text:
        documents = [doc.strip() for doc in context_text.split('\n\n') if doc.strip()]
    else:
        documents = [context_text]

    from document_intelligence import DocumentIntelligence
    di = DocumentIntelligence()

    try:
        di.analyze_document_corpus(documents)  # Initialize the corpus
        answer = di.answer_question(question, documents)

        return f"""
# ❓ Question Answering

**Question**: {question}
**Answer**: {answer['answer']}
**Confidence**: {answer['confidence']:.2f}
**Question Type**: {answer['question_type']}
"""
    except Exception as e:
        return f"Error answering question: {str(e)}"

def get_document_insights_interface(document_text):
    """Get comprehensive document insights"""
    if not document_text.strip():
        return "Please provide a document to analyze."

    from document_intelligence import DocumentIntelligence
    di = DocumentIntelligence()

    try:
        insights = di.generate_document_insights(document_text)

        result = f"""
# Document Insights

## Basic Statistics
- **Words**: {insights['basic_stats']['word_count']}
- **Sentences**: {insights['basic_stats']['sentence_count']}
- **Characters**: {insights['basic_stats']['character_count']}

## Key Phrases
"""

        for i, phrase in enumerate(insights['key_phrases'][:8], 1):
            result += f"{i}. **{phrase['phrase']}** (score: {phrase['score']:.2f})\n"

        result += f"""
## Summary
{insights['summary']['summary']}

## Analysis Scores
- **Readability**: {insights['readability']:.1f}
- **Complexity**: {insights['complexity_score']:.1f}
- **Sentiment**: {insights['sentiment']['sentiment']} ({insights['sentiment']['confidence']:.2f})
"""

        return result

    except Exception as e:
        return f"Error analyzing document: {str(e)}"

# =============================================================================
# GRADIO INTERFACE
# =============================================================================

def create_gradio_app():
    """Create the main Gradio application"""

    with gr.Blocks(title="Text Similarity Explorer", theme=gr.themes.Soft()) as app:

        gr.Markdown("""
        # Text Similarity Explorer
        ### A Visual Embeddings Playground using GTE-ModernBERT-Base
        
        This tool helps you explore text similarities using state-of-the-art embedding models and interactive visualizations.
        """)

        with gr.Tabs():

            # =================================================================
            # TAB 1: DATASET MANAGEMENT
            # =================================================================
            with gr.Tab("Dataset Management"):
                gr.Markdown("## Load and Manage Your Dataset")

                with gr.Row():
                    with gr.Column(scale=1):
                        dataset_choice = gr.Radio(
                            choices=["Pre-loaded Datasets", "Upload Custom File", "Enter Custom Text"],
                            value="Pre-loaded Datasets",
                            label="Choose Data Source"
                        )

                        # Pre-loaded datasets
                        preloaded_dropdown = gr.Dropdown(
                            choices=list(get_preloaded_datasets().keys()),
                            label="Select Pre-loaded Dataset",
                            visible=True
                        )
                        load_preloaded_btn = gr.Button("Load Dataset", variant="primary", visible=True)

                        # Custom file upload
                        custom_file = gr.File(
                            label="Upload CSV or JSON file",
                            file_types=[".csv", ".json"],
                            visible=False
                        )
                        load_file_btn = gr.Button("Load File", variant="primary", visible=False)

                        # Custom text input
                        custom_text_input = gr.Textbox(
                            label="Enter texts (one per line)",
                            lines=10,
                            placeholder="Enter your texts here, one per line...",
                            visible=False
                        )
                        process_text_btn = gr.Button("Process Texts", variant="primary", visible=False)

                    with gr.Column(scale=2):
                        dataset_status = gr.Textbox(
                            label="Dataset Status",
                            lines=10,
                            interactive=False
                        )

                        dataset_preview = gr.Dataframe(
                            label="Dataset Preview"
                        )

                # Generate embeddings section
                gr.Markdown("## Generate Embeddings")
                with gr.Row():
                    generate_btn = gr.Button("Generate Embeddings", variant="primary", size="lg")
                    embedding_status = gr.Textbox(
                        label="Embedding Status",
                        lines=8,
                        interactive=False
                    )

                # Event handlers for dataset choice
                def update_dataset_interface(choice):
                    if choice == "Pre-loaded Datasets":
                        return (
                            gr.update(visible=True), gr.update(visible=True),   # preloaded
                            gr.update(visible=False), gr.update(visible=False), # file
                            gr.update(visible=False), gr.update(visible=False)  # text
                        )
                    elif choice == "Upload Custom File":
                        return (
                            gr.update(visible=False), gr.update(visible=False), # preloaded
                            gr.update(visible=True), gr.update(visible=True),   # file
                            gr.update(visible=False), gr.update(visible=False)  # text
                        )
                    else:  # Enter Custom Text
                        return (
                            gr.update(visible=False), gr.update(visible=False), # preloaded
                            gr.update(visible=False), gr.update(visible=False), # file
                            gr.update(visible=True), gr.update(visible=True)    # text
                        )

                dataset_choice.change(
                    update_dataset_interface,
                    inputs=[dataset_choice],
                    outputs=[preloaded_dropdown, load_preloaded_btn, custom_file, load_file_btn, custom_text_input, process_text_btn]
                )

                # Event handlers for loading data
                load_preloaded_btn.click(
                    load_preloaded_dataset,
                    inputs=[preloaded_dropdown],
                    outputs=[dataset_status, dataset_preview]
                )

                load_file_btn.click(
                    load_custom_file,
                    inputs=[custom_file],
                    outputs=[dataset_status, dataset_preview]
                )

                process_text_btn.click(
                    process_custom_texts,
                    inputs=[custom_text_input],
                    outputs=[dataset_status, dataset_preview]
                )

                # Generate embeddings
                generate_btn.click(
                    generate_embeddings_func,
                    outputs=[embedding_status]
                )

            # =================================================================
            # TAB 2: SIMILARITY SEARCH
            # =================================================================
            with gr.Tab("Similarity Search"):
                gr.Markdown("## Semantic Similarity Search")

                with gr.Row():
                    with gr.Column(scale=1):
                        search_query = gr.Textbox(
                            label="Search Query",
                            lines=3,
                            placeholder="Enter text to find similar content..."
                        )

                        with gr.Row():
                            similarity_threshold = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.3,
                                step=0.05,
                                label="Similarity Threshold"
                            )
                            max_results = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=10,
                                step=1,
                                label="Max Results"
                            )

                        search_btn = gr.Button("Search Similar Texts", variant="primary")

                    with gr.Column(scale=2):
                        search_results = gr.Textbox(
                            label="Search Results Summary",
                            lines=15,
                            interactive=False
                        )

                search_results_table = gr.Dataframe(
                    label="Detailed Results"
                )

                search_btn.click(
                    search_similar_texts,
                    inputs=[search_query, similarity_threshold, max_results],
                    outputs=[search_results, search_results_table]
                )

            # =================================================================
            # TAB 3: VISUALIZATION
            # =================================================================
            with gr.Tab("Visualization"):
                gr.Markdown("## 2D Embedding Visualization")

                with gr.Row():
                    with gr.Column(scale=1):
                        color_by = gr.Dropdown(
                            choices=["category", "text_length", "none"],
                            value="category",
                            label="Color By"
                        )

                        point_size = gr.Slider(
                            minimum=3,
                            maximum=15,
                            value=8,
                            step=1,
                            label="Point Size"
                        )

                        opacity = gr.Slider(
                            minimum=0.3,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            label="Opacity"
                        )

                        update_viz_btn = gr.Button("Update Visualization", variant="primary")

                    with gr.Column(scale=3):
                        visualization_plot = gr.Plot(
                            label="UMAP 2D Projection"
                        )

                gr.Markdown("**Tip**: Points that are close together in the visualization represent texts with similar semantic meaning.")

                update_viz_btn.click(
                    create_visualization,
                    inputs=[color_by, point_size, opacity],
                    outputs=[visualization_plot]
                )

            # =================================================================
            # TAB 4: DATASET OVERVIEW & METRICS
            # =================================================================
            with gr.Tab("Overview & Metrics"):
                gr.Markdown("## Dataset Overview and Performance Metrics")

                with gr.Row():
                    refresh_overview_btn = gr.Button("Refresh Overview", variant="primary")
                    refresh_metrics_btn = gr.Button("Refresh Metrics", variant="primary")

                with gr.Row():
                    with gr.Column():
                        dataset_overview = gr.Markdown(
                            value="No dataset loaded",
                            label="Dataset Overview"
                        )

                    with gr.Column():
                        performance_metrics = gr.Markdown(
                            value="No embeddings generated",
                            label="Performance Metrics"
                        )

                refresh_overview_btn.click(
                    get_dataset_overview,
                    outputs=[dataset_overview]
                )

                refresh_metrics_btn.click(
                    get_performance_metrics,
                    outputs=[performance_metrics]
                )

            # =================================================================
            # TAB 5: DOCUMENT INTELLIGENCE
            # =================================================================
            with gr.Tab("Document Intelligence"):
                gr.Markdown("## Advanced Document Intelligence")

                with gr.Tabs():
                    # Sub-tab 1: Corpus Analysis
                    with gr.Tab("Corpus Analysis"):
                        gr.Markdown("### Analyze multiple documents for insights")

                        corpus_input = gr.Textbox(
                            label="Documents (separate with double newlines)",
                            lines=10,
                            placeholder="Document 1 text here...\n\nDocument 2 text here...\n\nDocument 3 text here..."
                        )

                        analyze_corpus_btn = gr.Button("Analyze Corpus", variant="primary")
                        corpus_results = gr.Markdown(label="Analysis Results")

                        analyze_corpus_btn.click(
                            analyze_corpus_intelligence,
                            inputs=[corpus_input],
                            outputs=[corpus_results]
                        )

                    # Sub-tab 2: Document Summarization
                    with gr.Tab("Summarization"):
                        gr.Markdown("### Generate intelligent summaries")

                        with gr.Row():
                            with gr.Column():
                                summary_input = gr.Textbox(
                                    label="Document to Summarize",
                                    lines=8,
                                    placeholder="Enter the document text you want to summarize..."
                                )

                                summary_length = gr.Slider(
                                    minimum=1,
                                    maximum=10,
                                    value=3,
                                    step=1,
                                    label="Summary Length (sentences)"
                                )

                                summarize_btn = gr.Button("Generate Summary", variant="primary")

                            with gr.Column():
                                summary_results = gr.Markdown(label="Summary Results")

                        summarize_btn.click(
                            summarize_document_interface,
                            inputs=[summary_input, summary_length],
                            outputs=[summary_results]
                        )

                    # Sub-tab 3: Question Answering
                    with gr.Tab("Q&A System"):
                        gr.Markdown("### Ask questions about your documents")

                        with gr.Row():
                            with gr.Column():
                                qa_question = gr.Textbox(
                                    label="Your Question",
                                    placeholder="What is this document about?"
                                )

                                qa_context = gr.Textbox(
                                    label="Context Documents",
                                    lines=8,
                                    placeholder="Paste your documents here for the AI to search through..."
                                )

                                qa_btn = gr.Button("Get Answer", variant="primary")

                            with gr.Column():
                                qa_results = gr.Markdown(label="Answer & Sources")

                        qa_btn.click(
                            answer_question_interface,
                            inputs=[qa_question, qa_context],
                            outputs=[qa_results]
                        )

                    # Sub-tab 4: Document Insights
                    with gr.Tab("Deep Insights"):
                        gr.Markdown("### Comprehensive document analysis")

                        with gr.Row():
                            with gr.Column():
                                insights_input = gr.Textbox(
                                    label="Document for Analysis",
                                    lines=10,
                                    placeholder="Enter a document for comprehensive analysis..."
                                )

                                insights_btn = gr.Button("Analyze Document", variant="primary")

                            with gr.Column():
                                insights_results = gr.Markdown(label="Document Insights")

                        insights_btn.click(
                            get_document_insights_interface,
                            inputs=[insights_input],
                            outputs=[insights_results]
                        )

            # =================================================================
            # TAB 6: EXPORT
            # =================================================================
            with gr.Tab("Export"):
                gr.Markdown("## Export Results")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Export Options")
                        include_embeddings = gr.Checkbox(
                            label="Include Embeddings",
                            value=False
                        )
                        include_umap = gr.Checkbox(
                            label="Include UMAP Coordinates",
                            value=True
                        )

                        export_btn = gr.Button("Export to CSV", variant="primary")

                        export_file = gr.File(
                            label="Download File",
                            visible=False
                        )

                    with gr.Column():
                        gr.Markdown("### Export Information")
                        gr.Markdown("""
                        **Available Export Options:**
                        - **Dataset Only**: Basic text data and metadata
                        - **With Embeddings**: Include high-dimensional embedding vectors
                        - **With UMAP**: Include 2D coordinates for visualization
                        
                        **File Format:** CSV (Comma-separated values)
                        **Compatibility:** Excel, Google Sheets, Python, R
                        """)

                export_btn.click(
                    export_data,
                    inputs=[include_embeddings, include_umap],
                    outputs=[export_file]
                )

        # Footer
        gr.Markdown("""
        ---
        **Model:** Alibaba-NLP/gte-modernbert-base | **Framework:** Sentence Transformers | **Visualization:** UMAP
        """)

    return app

# =============================================================================
# LAUNCH APPLICATION
# =============================================================================

if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(
        share=False,
        debug=True,
        server_name="127.0.0.1",
        server_port=7860
    )