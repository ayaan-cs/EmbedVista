"""
Document Intelligence Module
===========================

Advanced NLP capabilities for semantic document analysis including:
- Key phrase extraction with TF-IDF and semantic analysis
- Intelligent document summarization using extractive and abstractive methods
- Question-answering system for document corpus
- Automatic document classification and tagging
- Semantic document comparison and ranking

Author: AI-Powered Document Intelligence System
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import re
from collections import Counter, defaultdict
import logging
from datetime import datetime

# NLP Libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from nltk.stem import WordNetLemmatizer
    from nltk.util import ngrams
except ImportError as e:
    print(f"Warning: Some NLP libraries not available: {e}")
    print("Install with: pip install scikit-learn nltk")

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_sent_tokenize(text):
    """Safe sentence tokenization with fallback"""
    try:
        return sent_tokenize(text)
    except:
        # Simple fallback - split on sentence endings
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

def safe_word_tokenize(text):
    """Safe word tokenization with fallback"""
    try:
        return word_tokenize(text)
    except:
        # Simple fallback - split on whitespace and punctuation
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        return words

class DocumentIntelligence:
    """
    Advanced document intelligence system for semantic analysis and insights.

    This class provides comprehensive document analysis capabilities including
    key phrase extraction, summarization, question-answering, and classification.
    """

    def __init__(self, embedding_generator=None):
        """
        Initialize the Document Intelligence system.

        Args:
            embedding_generator: Pre-trained embedding generator for semantic analysis
        """
        self.embedding_generator = embedding_generator
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = None
        self.document_embeddings = None
        self.documents = None
        self.document_metadata = {}

        logger.info("Document Intelligence system initialized")

    def analyze_document_corpus(self, documents: List[str], metadata: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on a document corpus.

        Args:
            documents: List of document texts
            metadata: Optional metadata for each document

        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Analyzing corpus of {len(documents)} documents")

        self.documents = documents
        if metadata:
            self.document_metadata = {i: meta for i, meta in enumerate(metadata)}

        # Generate document embeddings if embedding generator is available
        if self.embedding_generator:
            try:
                self.document_embeddings = self.embedding_generator.generate_embeddings(documents)
                logger.info("Document embeddings generated successfully")
            except Exception as e:
                logger.warning(f"Failed to generate embeddings: {e}")
                self.document_embeddings = None

        # TF-IDF analysis
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3),
            max_df=0.8,
            min_df=2
        )

        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            logger.info("TF-IDF analysis completed")
        except Exception as e:
            logger.error(f"TF-IDF analysis failed: {e}")
            tfidf_matrix = None

        # Comprehensive analysis
        analysis_results = {
            'corpus_stats': self._get_corpus_statistics(documents),
            'key_phrases': self.extract_key_phrases(documents),
            'topics': self._discover_topics(documents),
            'document_similarities': self._calculate_document_similarities(),
            'readability_scores': self._calculate_readability_scores(documents),
            'sentiment_analysis': self._analyze_sentiment(documents),
            'named_entities': self._extract_named_entities(documents),
            'analysis_timestamp': datetime.now().isoformat()
        }

        logger.info("Corpus analysis completed successfully")
        return analysis_results

    def extract_key_phrases(self, documents: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Extract key phrases using multiple techniques.

        Args:
            documents: List of document texts
            top_k: Number of top phrases to return

        Returns:
            List of key phrases with scores and metadata
        """
        logger.info(f"Extracting top {top_k} key phrases")

        all_phrases = []
        phrase_scores = defaultdict(float)
        phrase_docs = defaultdict(set)

        for doc_idx, doc in enumerate(documents):
            # Extract n-grams (1-3 words)
            doc_phrases = self._extract_ngram_phrases(doc, n_range=(1, 3))

            # Score phrases using multiple methods
            for phrase, score in doc_phrases.items():
                phrase_scores[phrase] += score
                phrase_docs[phrase].add(doc_idx)

        # Calculate final scores incorporating document frequency
        final_phrases = []
        for phrase, score in phrase_scores.items():
            if len(phrase.split()) > 1 or len(phrase) > 3:  # Filter out short single words
                doc_frequency = len(phrase_docs[phrase])
                final_score = score * (1 + np.log(doc_frequency))

                final_phrases.append({
                    'phrase': phrase,
                    'score': final_score,
                    'document_frequency': doc_frequency,
                    'documents': list(phrase_docs[phrase]),
                    'importance': self._categorize_importance(final_score, len(documents))
                })

        # Sort by score and return top_k
        final_phrases.sort(key=lambda x: x['score'], reverse=True)
        return final_phrases[:top_k]

    def summarize_document(self, document: str, summary_length: int = 3) -> Dict[str, Any]:
        """
        Generate extractive summary of a document.

        Args:
            document: Document text to summarize
            summary_length: Number of sentences in summary

        Returns:
            Dictionary containing summary and metadata
        """
        logger.info(f"Generating {summary_length}-sentence summary")

        # Split into sentences
        sentences = sent_tokenize(document)

        if len(sentences) <= summary_length:
            return {
                'summary': document,
                'sentences': sentences,
                'compression_ratio': 1.0,
                'method': 'no_compression_needed'
            }

        # Calculate sentence scores
        sentence_scores = self._calculate_sentence_scores(sentences, document)

        # Select top sentences while preserving order
        top_sentences = sorted(sentence_scores, key=lambda x: x['score'], reverse=True)[:summary_length]
        top_sentences.sort(key=lambda x: x['index'])  # Preserve original order

        summary_text = ' '.join([sent['sentence'] for sent in top_sentences])

        return {
            'summary': summary_text,
            'sentences': [sent['sentence'] for sent in top_sentences],
            'sentence_scores': sentence_scores,
            'compression_ratio': len(summary_text) / len(document),
            'method': 'extractive_tfidf',
            'original_length': len(sentences),
            'summary_length': summary_length
        }

    def answer_question(self, question: str, context_documents: Optional[List[str]] = None, top_k: int = 5) -> Dict[str, Any]:
        """
        Answer questions based on document corpus using semantic similarity.

        Args:
            question: Question to answer
            context_documents: Optional specific documents to search in
            top_k: Number of relevant passages to consider

        Returns:
            Dictionary containing answer and supporting information
        """
        logger.info(f"Answering question: '{question[:50]}...'")

        if context_documents is None:
            context_documents = self.documents

        if not context_documents:
            return {
                'answer': "No documents available to answer the question.",
                'confidence': 0.0,
                'sources': []
            }

        # Find most relevant documents/passages
        relevant_passages = self._find_relevant_passages(question, context_documents, top_k)

        if not relevant_passages:
            return {
                'answer': "No relevant information found to answer this question.",
                'confidence': 0.0,
                'sources': []
            }

        # Generate answer from most relevant passage
        best_passage = relevant_passages[0]
        answer = self._extract_answer_from_passage(question, best_passage['text'])

        return {
            'answer': answer['text'],
            'confidence': answer['confidence'],
            'sources': relevant_passages,
            'question_type': self._classify_question_type(question),
            'answer_method': 'semantic_similarity'
        }

    def classify_documents(self, documents: List[str], categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Classify documents into categories using semantic analysis.

        Args:
            documents: List of documents to classify
            categories: Optional predefined categories

        Returns:
            List of classification results for each document
        """
        logger.info(f"Classifying {len(documents)} documents")

        if not categories:
            # Auto-discover categories using clustering
            categories = self._auto_discover_categories(documents)

        classifications = []

        for doc_idx, doc in enumerate(documents):
            # Classify using semantic similarity
            classification = self._classify_single_document(doc, categories)

            classifications.append({
                'document_index': doc_idx,
                'document_preview': doc[:100] + '...',
                'predicted_category': classification['category'],
                'confidence': classification['confidence'],
                'all_scores': classification['scores'],
                'features': self._extract_classification_features(doc)
            })

        return classifications

    def find_similar_documents(self, query_doc: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find documents similar to a query document.

        Args:
            query_doc: Document to find similarities for
            top_k: Number of similar documents to return

        Returns:
            List of similar documents with similarity scores
        """
        if not self.documents:
            return []

        if self.document_embeddings is not None and self.embedding_generator:
            # Use semantic embeddings
            query_embedding = self.embedding_generator.encode_single(query_doc)
            similarities = cosine_similarity([query_embedding], self.document_embeddings)[0]
        else:
            # Fallback to TF-IDF
            if self.tfidf_vectorizer is None:
                self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
                self.tfidf_vectorizer.fit(self.documents)

            query_tfidf = self.tfidf_vectorizer.transform([query_doc])
            doc_tfidf = self.tfidf_vectorizer.transform(self.documents)
            similarities = cosine_similarity(query_tfidf, doc_tfidf)[0]

        # Get top similar documents
        similar_indices = np.argsort(similarities)[::-1][:top_k]

        similar_docs = []
        for idx in similar_indices:
            similar_docs.append({
                'document_index': int(idx),
                'document_text': self.documents[idx],
                'similarity_score': float(similarities[idx]),
                'metadata': self.document_metadata.get(idx, {}),
                'preview': self.documents[idx][:200] + '...'
            })

        return similar_docs

    def generate_document_insights(self, document: str) -> Dict[str, Any]:
        """
        Generate comprehensive insights for a single document.

        Args:
            document: Document text to analyze

        Returns:
            Dictionary containing various insights
        """
        insights = {
            'basic_stats': {
                'word_count': len(document.split()),
                'sentence_count': len(sent_tokenize(document)),
                'character_count': len(document),
                'average_sentence_length': np.mean([len(sent.split()) for sent in sent_tokenize(document)])
            },
            'key_phrases': self.extract_key_phrases([document], top_k=10),
            'summary': self.summarize_document(document),
            'readability': self._calculate_readability_score(document),
            'sentiment': self._analyze_document_sentiment(document),
            'named_entities': self._extract_document_entities(document),
            'complexity_score': self._calculate_complexity_score(document),
            'topics': self._extract_document_topics(document)
        }

        return insights

    # Private helper methods

    def _get_corpus_statistics(self, documents: List[str]) -> Dict[str, Any]:
        """Calculate basic statistics for the document corpus."""
        word_counts = [len(doc.split()) for doc in documents]
        sentence_counts = [len(sent_tokenize(doc)) for doc in documents]

        return {
            'document_count': len(documents),
            'total_words': sum(word_counts),
            'average_words_per_document': np.mean(word_counts),
            'total_sentences': sum(sentence_counts),
            'average_sentences_per_document': np.mean(sentence_counts),
            'vocabulary_size': len(set(' '.join(documents).lower().split())),
            'longest_document': max(word_counts),
            'shortest_document': min(word_counts)
        }

    def _extract_ngram_phrases(self, text: str, n_range: Tuple[int, int] = (1, 3)) -> Dict[str, float]:
        """Extract n-gram phrases with TF-IDF-like scoring."""
        # Clean and tokenize
        words = word_tokenize(text.lower())
        words = [self.lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in self.stop_words]

        phrase_scores = {}

        for n in range(n_range[0], n_range[1] + 1):
            for ngram in ngrams(words, n):
                phrase = ' '.join(ngram)
                # Simple scoring based on word frequency and phrase length
                phrase_scores[phrase] = phrase_scores.get(phrase, 0) + (n * 0.5)

        return phrase_scores

    def _categorize_importance(self, score: float, total_docs: int) -> str:
        """Categorize phrase importance based on score."""
        if score > total_docs * 2:
            return 'high'
        elif score > total_docs:
            return 'medium'
        else:
            return 'low'

    def _calculate_sentence_scores(self, sentences: List[str], document: str) -> List[Dict[str, Any]]:
        """Calculate importance scores for sentences."""
        # Simple TF-IDF based sentence scoring
        vectorizer = TfidfVectorizer(stop_words='english')

        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sentence_scores = []

            for i, sentence in enumerate(sentences):
                # Score based on TF-IDF sum and position
                tfidf_score = np.sum(tfidf_matrix[i].toarray())
                position_score = 1.0 / (i + 1) if i < 3 else 0.5  # Favor early sentences

                sentence_scores.append({
                    'sentence': sentence,
                    'score': tfidf_score + position_score,
                    'index': i,
                    'tfidf_score': tfidf_score,
                    'position_score': position_score
                })

            return sentence_scores
        except:
            # Fallback to simple length-based scoring
            return [{'sentence': sent, 'score': len(sent.split()), 'index': i}
                    for i, sent in enumerate(sentences)]

    def _find_relevant_passages(self, question: str, documents: List[str], top_k: int) -> List[Dict[str, Any]]:
        """Find passages relevant to a question."""
        if self.document_embeddings is not None and self.embedding_generator:
            # Use semantic similarity
            question_embedding = self.embedding_generator.encode_single(question)
            similarities = cosine_similarity([question_embedding], self.document_embeddings)[0]
        else:
            # Fallback to keyword matching
            question_words = set(question.lower().split()) - self.stop_words
            similarities = []

            for doc in documents:
                doc_words = set(doc.lower().split())
                overlap = len(question_words & doc_words)
                similarities.append(overlap / len(question_words) if question_words else 0)

            similarities = np.array(similarities)

        # Get top passages
        top_indices = np.argsort(similarities)[::-1][:top_k]

        passages = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum relevance threshold
                passages.append({
                    'text': documents[idx],
                    'similarity': float(similarities[idx]),
                    'document_index': int(idx),
                    'preview': documents[idx][:200] + '...'
                })

        return passages

    def _extract_answer_from_passage(self, question: str, passage: str) -> Dict[str, Any]:
        """Extract answer from a relevant passage."""
        # Simple extractive approach
        sentences = sent_tokenize(passage)

        question_words = set(question.lower().split()) - self.stop_words

        best_sentence = ""
        best_score = 0

        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words & sentence_words)
            score = overlap / len(question_words) if question_words else 0

            if score > best_score:
                best_score = score
                best_sentence = sentence

        return {
            'text': best_sentence if best_sentence else sentences[0] if sentences else "No answer found.",
            'confidence': min(best_score, 1.0)
        }

    def _classify_question_type(self, question: str) -> str:
        """Classify the type of question."""
        question_lower = question.lower()

        if any(word in question_lower for word in ['what', 'which', 'who']):
            return 'factual'
        elif any(word in question_lower for word in ['how', 'why']):
            return 'explanatory'
        elif any(word in question_lower for word in ['when', 'where']):
            return 'temporal_spatial'
        else:
            return 'general'

    def _discover_topics(self, documents: List[str], n_topics: int = 5) -> List[Dict[str, Any]]:
        """Discover topics using simple clustering."""
        if len(documents) < n_topics:
            n_topics = len(documents)

        try:
            # Use TF-IDF for topic discovery
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(documents)

            # K-means clustering
            kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(tfidf_matrix)

            # Extract topic keywords
            feature_names = vectorizer.get_feature_names_out()
            topics = []

            for i in range(n_topics):
                # Get top words for this cluster
                cluster_center = kmeans.cluster_centers_[i]
                top_indices = cluster_center.argsort()[::-1][:10]
                top_words = [feature_names[idx] for idx in top_indices]

                # Count documents in cluster
                doc_count = np.sum(clusters == i)

                topics.append({
                    'topic_id': i,
                    'keywords': top_words,
                    'document_count': int(doc_count),
                    'strength': float(np.max(cluster_center))
                })

            return topics
        except:
            return []

    def _calculate_document_similarities(self) -> Dict[str, Any]:
        """Calculate similarity matrix for documents."""
        if not self.documents or len(self.documents) < 2:
            return {}

        if self.document_embeddings is not None:
            similarity_matrix = cosine_similarity(self.document_embeddings)
        else:
            # Fallback to TF-IDF
            if self.tfidf_vectorizer is None:
                self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')

            tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)
            similarity_matrix = cosine_similarity(tfidf_matrix)

        # Find most and least similar pairs
        n_docs = len(self.documents)
        similarities = []

        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                similarities.append({
                    'doc1_index': i,
                    'doc2_index': j,
                    'similarity': float(similarity_matrix[i, j])
                })

        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        return {
            'average_similarity': float(np.mean(similarity_matrix[np.triu_indices(n_docs, k=1)])),
            'most_similar_pairs': similarities[:5],
            'least_similar_pairs': similarities[-5:],
            'similarity_matrix_shape': similarity_matrix.shape
        }

    def _calculate_readability_scores(self, documents: List[str]) -> List[Dict[str, Any]]:
        """Calculate readability scores for documents."""
        scores = []

        for i, doc in enumerate(documents):
            score = self._calculate_readability_score(doc)
            scores.append({
                'document_index': i,
                'readability_score': score,
                'readability_level': self._categorize_readability(score)
            })

        return scores

    def _calculate_readability_score(self, text: str) -> float:
        """Calculate simple readability score (Flesch-like)."""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)

        if not sentences or not words:
            return 0.0

        avg_sentence_length = len(words) / len(sentences)
        # Simple approximation
        readability = max(0, 100 - avg_sentence_length * 2)

        return readability

    def _categorize_readability(self, score: float) -> str:
        """Categorize readability score."""
        if score >= 80:
            return 'very_easy'
        elif score >= 60:
            return 'easy'
        elif score >= 40:
            return 'moderate'
        elif score >= 20:
            return 'difficult'
        else:
            return 'very_difficult'

    def _analyze_sentiment(self, documents: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment of documents (simple approach)."""
        sentiments = []

        for i, doc in enumerate(documents):
            sentiment = self._analyze_document_sentiment(doc)
            sentiments.append({
                'document_index': i,
                **sentiment
            })

        return sentiments

    def _analyze_document_sentiment(self, document: str) -> Dict[str, Any]:
        """Analyze sentiment of a single document."""
        # Simple lexicon-based approach
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'best'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disappointing', 'poor'}

        words = word_tokenize(document.lower())

        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        if positive_count > negative_count:
            sentiment = 'positive'
            confidence = (positive_count - negative_count) / len(words)
        elif negative_count > positive_count:
            sentiment = 'negative'
            confidence = (negative_count - positive_count) / len(words)
        else:
            sentiment = 'neutral'
            confidence = 0.0

        return {
            'sentiment': sentiment,
            'confidence': min(confidence, 1.0),
            'positive_words': positive_count,
            'negative_words': negative_count
        }

    def _extract_named_entities(self, documents: List[str]) -> List[Dict[str, Any]]:
        """Extract named entities from documents."""
        entities_per_doc = []

        for i, doc in enumerate(documents):
            entities = self._extract_document_entities(doc)
            entities_per_doc.append({
                'document_index': i,
                'entities': entities
            })

        return entities_per_doc

    def _extract_document_entities(self, document: str) -> List[Dict[str, str]]:
        """Extract named entities from a single document."""
        try:
            # Tokenize and tag
            tokens = word_tokenize(document)
            pos_tags = pos_tag(tokens)

            # Named entity recognition
            entities = []
            chunks = ne_chunk(pos_tags, binary=False)

            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    entity_name = ' '.join([token for token, pos in chunk.leaves()])
                    entities.append({
                        'text': entity_name,
                        'type': chunk.label()
                    })

            return entities
        except:
            # Simple fallback - find capitalized words
            words = word_tokenize(document)
            entities = []

            for word in words:
                if word.istitle() and len(word) > 2:
                    entities.append({
                        'text': word,
                        'type': 'UNKNOWN'
                    })

            return entities[:10]  # Limit to avoid too many results

    def _auto_discover_categories(self, documents: List[str], n_categories: int = 5) -> List[str]:
        """Auto-discover document categories using clustering."""
        topics = self._discover_topics(documents, n_categories)

        categories = []
        for topic in topics:
            # Create category name from top keywords
            category_name = '_'.join(topic['keywords'][:2])
            categories.append(category_name)

        return categories if categories else ['general']

    def _classify_single_document(self, document: str, categories: List[str]) -> Dict[str, Any]:
        """Classify a single document into predefined categories."""
        # Simple keyword-based classification
        doc_words = set(document.lower().split())

        scores = {}
        for category in categories:
            category_words = set(category.lower().replace('_', ' ').split())
            overlap = len(doc_words & category_words)
            scores[category] = overlap / len(category_words) if category_words else 0

        best_category = max(scores, key=scores.get) if scores else categories[0]
        confidence = scores.get(best_category, 0)

        return {
            'category': best_category,
            'confidence': min(confidence, 1.0),
            'scores': scores
        }

    def _extract_classification_features(self, document: str) -> Dict[str, Any]:
        """Extract features used for classification."""
        return {
            'word_count': len(document.split()),
            'sentence_count': len(sent_tokenize(document)),
            'avg_word_length': np.mean([len(word) for word in document.split()]),
            'has_questions': '?' in document,
            'has_numbers': any(char.isdigit() for char in document),
            'uppercase_ratio': sum(1 for c in document if c.isupper()) / len(document)
        }

    def _calculate_complexity_score(self, document: str) -> float:
        """Calculate document complexity score."""
        words = word_tokenize(document)
        sentences = sent_tokenize(document)

        if not words or not sentences:
            return 0.0

        # Factors affecting complexity
        avg_word_length = np.mean([len(word) for word in words])
        avg_sentence_length = len(words) / len(sentences)
        unique_words_ratio = len(set(words)) / len(words)

        # Simple complexity score
        complexity = (avg_word_length * 0.3 +
                      avg_sentence_length * 0.4 +
                      unique_words_ratio * 30)

        return min(complexity, 100)

    def _extract_document_topics(self, document: str) -> List[str]:
        """Extract topics from a single document."""
        # Use key phrases as topics
        key_phrases = self.extract_key_phrases([document], top_k=5)
        return [phrase['phrase'] for phrase in key_phrases if phrase['score'] > 1.0]


# Utility functions for integration with main app

def create_document_intelligence_interface(embedding_generator=None):
    """
    Create a document intelligence instance ready for integration.

    Args:
        embedding_generator: Embedding generator from main app

    Returns:
        Configured DocumentIntelligence instance
    """
    return DocumentIntelligence(embedding_generator)

def analyze_dataset_with_intelligence(documents: List[str], metadata: Optional[List[Dict]] = None,
                                      embedding_generator=None) -> Dict[str, Any]:
    """
    Convenience function to analyze a dataset with document intelligence.

    Args:
        documents: List of documents to analyze
        metadata: Optional metadata for documents
        embedding_generator: Embedding generator instance

    Returns:
        Comprehensive analysis results
    """
    di = DocumentIntelligence(embedding_generator)
    return di.analyze_document_corpus(documents, metadata)

def demo_document_intelligence():
    """
    Demo function to showcase document intelligence capabilities.

    Returns:
        Demo results showing various features
    """
    # Sample documents for demonstration
    sample_docs = [
        "Artificial intelligence is revolutionizing the way we process and analyze data. Machine learning algorithms can now identify patterns in vast datasets that would be impossible for humans to detect manually.",

        "Climate change represents one of the most significant challenges facing humanity today. Rising global temperatures, melting ice caps, and extreme weather patterns are clear indicators of environmental disruption.",

        "The development of quantum computing promises to solve complex problems that are currently intractable with classical computers. Quantum algorithms could revolutionize cryptography, optimization, and scientific simulation.",

        "Modern medicine has made tremendous advances in treating diseases that were once considered incurable. Gene therapy, immunotherapy, and precision medicine are opening new frontiers in healthcare.",

        "Space exploration continues to push the boundaries of human knowledge and technological capability. Recent missions to Mars and the development of reusable rockets are making space more accessible than ever before."
    ]

    # Create document intelligence instance
    di = DocumentIntelligence()

    print("Document Intelligence Demo")
    print("=" * 50)

    # Analyze the corpus
    print("\nAnalyzing document corpus...")
    analysis = di.analyze_document_corpus(sample_docs)

    print(f"Analyzed {analysis['corpus_stats']['document_count']} documents")
    print(f"Total words: {analysis['corpus_stats']['total_words']}")
    print(f"Vocabulary size: {analysis['corpus_stats']['vocabulary_size']}")

    # Show key phrases
    print("\nKey Phrases Discovered:")
    for i, phrase in enumerate(analysis['key_phrases'][:5], 1):
        print(f"  {i}. {phrase['phrase']} (score: {phrase['score']:.2f})")

    # Show topics
    print(f"\nTopics Discovered ({len(analysis['topics'])}):")
    for i, topic in enumerate(analysis['topics'], 1):
        keywords = ', '.join(topic['keywords'][:5])
        print(f"  {i}. {keywords} ({topic['document_count']} docs)")

    # Demonstrate summarization
    print("\nDocument Summarization Demo:")
    long_doc = sample_docs[0] + " " + sample_docs[2]  # Combine two docs
    summary = di.summarize_document(long_doc, summary_length=2)
    print(f"Original length: {len(long_doc)} chars")
    print(f"Summary length: {len(summary['summary'])} chars")
    print(f"Compression ratio: {summary['compression_ratio']:.2f}")
    print(f"Summary: {summary['summary']}")

    # Demonstrate question answering
    print("\nQuestion Answering Demo:")
    questions = [
        "What is artificial intelligence?",
        "What are the challenges of climate change?",
        "How can quantum computing help with cryptography?"
    ]

    for question in questions:
        answer = di.answer_question(question, sample_docs)
        print(f"Q: {question}")
        print(f"A: {answer['answer']}")
        print(f"Confidence: {answer['confidence']:.2f}")
        print()

    # Demonstrate document insights
    print("\nDocument Insights Demo:")
    insights = di.generate_document_insights(sample_docs[0])
    print(f"Word count: {insights['basic_stats']['word_count']}")
    print(f"Readability score: {insights['readability']:.1f}")
    print(f"Sentiment: {insights['sentiment']['sentiment']} ({insights['sentiment']['confidence']:.2f})")
    print(f"Complexity score: {insights['complexity_score']:.1f}")

    return analysis

# Integration helper for Gradio interface
def create_gradio_intelligence_interface():
    """
    Create Gradio interface components for document intelligence features.

    Returns:
        Dictionary of Gradio interface functions
    """

    def analyze_corpus_interface(documents_text: str, embedding_generator=None):
        """Interface function for corpus analysis."""
        if not documents_text.strip():
            return "Please provide documents to analyze."

        # Split documents by double newline or numbered format
        if '\n\n' in documents_text:
            documents = [doc.strip() for doc in documents_text.split('\n\n') if doc.strip()]
        else:
            # Try to detect numbered format
            lines = documents_text.split('\n')
            documents = []
            current_doc = ""

            for line in lines:
                if line.strip() and (line[0].isdigit() or line.startswith('- ')):
                    if current_doc:
                        documents.append(current_doc.strip())
                    current_doc = line
                else:
                    current_doc += " " + line

            if current_doc:
                documents.append(current_doc.strip())

        if not documents:
            documents = [documents_text]  # Treat as single document

        # Analyze documents
        di = DocumentIntelligence(embedding_generator)
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
            result += f"{i}. **{phrase['phrase']}** (score: {phrase['score']:.2f}, docs: {phrase['document_frequency']})\n"

        result += f"""
## Discovered Topics
"""

        for i, topic in enumerate(analysis['topics'], 1):
            keywords = ', '.join(topic['keywords'][:8])
            result += f"{i}. **{keywords}** ({topic['document_count']} documents)\n"

        result += f"""
## Document Similarities
- **Average Similarity**: {analysis['document_similarities'].get('average_similarity', 0):.3f}

## Readability Analysis
"""

        for score in analysis['readability_scores'][:5]:
            result += f"- Doc {score['document_index'] + 1}: {score['readability_score']:.1f} ({score['readability_level']})\n"

        return result

    def summarize_interface(document_text: str, summary_length: int = 3):
        """Interface function for document summarization."""
        if not document_text.strip():
            return "Please provide a document to summarize."

        di = DocumentIntelligence()
        summary = di.summarize_document(document_text, summary_length)

        return f"""
# Document Summary

**Original Length**: {len(document_text)} characters ({summary['original_length']} sentences)
**Summary Length**: {len(summary['summary'])} characters ({summary['summary_length']} sentences)
**Compression Ratio**: {summary['compression_ratio']:.2f}

## Summary:
{summary['summary']}

## Selected Sentences:
"""  + '\n'.join([f"{i+1}. {sent}" for i, sent in enumerate(summary['sentences'])])

    def question_answer_interface(question: str, context_text: str):
        """Interface function for question answering."""
        if not question.strip() or not context_text.strip():
            return "Please provide both a question and context documents."

        # Split context into documents
        if '\n\n' in context_text:
            documents = [doc.strip() for doc in context_text.split('\n\n') if doc.strip()]
        else:
            documents = [context_text]

        di = DocumentIntelligence()
        di.analyze_document_corpus(documents)  # Initialize the corpus

        answer = di.answer_question(question, documents)

        result = f"""
# Question Answering

**Question**: {question}
**Answer**: {answer['answer']}
**Confidence**: {answer['confidence']:.2f}
**Question Type**: {answer['question_type']}

## Supporting Sources:
"""

        for i, source in enumerate(answer['sources'][:3], 1):
            result += f"{i}. Similarity: {source['similarity']:.3f}\n   {source['preview']}\n\n"

        return result

    def document_insights_interface(document_text: str):
        """Interface function for document insights."""
        if not document_text.strip():
            return "Please provide a document to analyze."

        di = DocumentIntelligence()
        insights = di.generate_document_insights(document_text)

        result = f"""
# Document Insights

## Basic Statistics
- **Words**: {insights['basic_stats']['word_count']}
- **Sentences**: {insights['basic_stats']['sentence_count']}
- **Characters**: {insights['basic_stats']['character_count']}
- **Avg Sentence Length**: {insights['basic_stats']['average_sentence_length']:.1f} words

## Key Phrases
"""

        for i, phrase in enumerate(insights['key_phrases'][:8], 1):
            result += f"{i}. **{phrase['phrase']}** (score: {phrase['score']:.2f})\n"

        result += f"""
## Summary
{insights['summary']['summary']}

## Analysis Scores
- **Readability**: {insights['readability']:.1f} ({insights.get('readability_level', 'N/A')})
- **Complexity**: {insights['complexity_score']:.1f}
- **Sentiment**: {insights['sentiment']['sentiment']} (confidence: {insights['sentiment']['confidence']:.2f})

##  Named Entities
"""

        entities = insights['named_entities'][:10]
        if entities:
            for entity in entities:
                result += f"- **{entity['text']}** ({entity['type']})\n"
        else:
            result += "No named entities detected.\n"

        result += f"""
## Topics
"""

        for i, topic in enumerate(insights['topics'][:5], 1):
            result += f"{i}. {topic}\n"

        return result

    return {
        'analyze_corpus': analyze_corpus_interface,
        'summarize_document': summarize_interface,
        'answer_question': question_answer_interface,
        'document_insights': document_insights_interface
    }


if __name__ == "__main__":
    # Run demo when script is executed directly
    demo_document_intelligence()