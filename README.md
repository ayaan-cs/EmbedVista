# 🔍 Text Similarity Explorer

> **Advanced Semantic Analysis Platform** powered by Modern Transformer Models

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)](https://gradio.app)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen.svg)]()

**A comprehensive, enterprise-grade platform for semantic text analysis, similarity search, and document intelligence.** Built with cutting-edge AI models and designed for both researchers and professionals who need powerful text analysis capabilities.

---

## **Features**

### **Core Capabilities**
- **Semantic Similarity Search** - Find related texts using state-of-the-art embeddings
- **Interactive 2D Visualizations** - Explore your data in UMAP-projected embedding space
- **Document Intelligence** - Advanced NLP analysis including summarization, Q&A, and insights
- **Multi-Dataset Support** - Pre-loaded datasets + custom file upload (CSV, JSON)
- **Real-time Analysis** - Live processing with performance monitoring

### **Advanced Document Intelligence**
- **Smart Summarization** - Extractive summarization with sentence scoring
- **Question Answering** - Semantic Q&A system for document exploration
- **Key Phrase Extraction** - Multi-method phrase extraction with importance scoring
- **Topic Discovery** - Automatic topic modeling and categorization
- **Sentiment Analysis** - Document-level sentiment with confidence scores
- **Readability Analysis** - Complexity scoring and reading level assessment

### **Professional Analytics**
- **Performance Monitoring** - Real-time metrics and system analytics
- **Export Capabilities** - CSV, JSON, Excel exports with metadata
- **Batch Processing** - Handle multiple documents efficiently
- **Comprehensive Reporting** - Detailed analysis reports and insights

---

## **Technology Stack**

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | Python 3.8+ | Core processing engine |
| **UI Framework** | Gradio | Interactive web interface |
| **ML Models** | Sentence Transformers | Text embeddings |
| **Primary Model** | Alibaba-NLP/gte-modernbert-base | State-of-the-art embeddings |
| **Fallback Model** | all-MiniLM-L6-v2 | Lightweight alternative |
| **Visualization** | UMAP, Plotly | 2D projections and charts |
| **NLP Processing** | NLTK, scikit-learn | Text analysis and clustering |
| **Data Handling** | Pandas, NumPy | Data manipulation |

---

##  **Installation**

### **Prerequisites**
- Python 3.8 or higher
- 4GB+ RAM recommended
- GPU support optional (CUDA-compatible)

### **Quick Setup**
```bash
# Clone the repository
git clone https://github.com/ayaan-cs/EmbedVista.git
cd EmbedVista

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python setup_nltk.py

# Launch the application
python gradio_app.py
```

### **Requirements**
```
requests~=2.32.3
pandas~=2.3.1
bs4~=0.0.2
beautifulsoup4~=4.13.4
gradio~=5.39.0
numpy~=2.2.6
plotly~=6.2.0
torch~=2.7.1
umap-learn~=0.5.9.post2
psutil~=7.0.0
sentence-transformers~=5.0.0
scikit-learn~=1.7.1
nltk~=3.9.1
streamlit~=1.47.1
datasets~=4.0.0
```

---

## 🎮 **Usage**

### **Quick Start**
1. **Launch the application**: `python gradio_app.py`
2. **Open your browser**: Navigate to `http://localhost:7860`
3. **Load data**: Choose from pre-loaded datasets or upload your own
4. **Generate embeddings**: Click "Generate Embeddings" to process your data
5. **Explore**: Use similarity search, visualizations, and document intelligence

### **Supported Data Formats**
- **CSV Files**: Must contain a 'text' column
- **JSON Files**: Array of objects with 'text' field
- **Direct Input**: Paste text directly (one text per line)

### **Pre-loaded Datasets**
-  **Movie Plot Summaries** - Film descriptions across genres
-  **Book Descriptions** - Literary summaries and reviews
-  **Product Reviews** - Customer feedback across categories
-  **News Headlines** - Current events and topics
-  **Scientific Abstracts** - Research paper summaries

---

##  **Roadmap & Upcoming Features**

### **Phase 1: Advanced Analytics** *(Q3 2025)*
- **Multi-Model Comparison** - Compare embeddings across different transformer models
- **Advanced Clustering** - DBSCAN, K-means, and hierarchical clustering with interactive exploration
- **Enhanced Performance** - GPU acceleration and optimized processing pipelines

### **Phase 2: Enhanced Visualizations** *(Q3-Q4 2025)*
- **3D Visualizations** - Interactive 3D embedding spaces with Three.js
- **Network Analysis** - Document relationship graphs and community detection
- **Advanced Charts** - Heatmaps, dendrograms, and statistical visualizations

### **Phase 3: Enterprise Features** *(Q4 2025)*
- **API Integrations** - RESTful API for programmatic access
- **Cloud Connectors** - Google Drive, Dropbox, AWS S3 integration
- **Collaboration Tools** - Multi-user support and shared workspaces
- **Advanced Export** - Professional reporting and dashboard integration

### **Phase 4: Production Deployment** *(Q1 2026)*
- **Standalone Application** - Desktop app with offline capabilities
- **Cloud Deployment** - Hosted SaaS version with enterprise features
- **Mobile Optimization** - Progressive Web App (PWA) support
- **Enterprise Security** - SSO, audit logs, and compliance features

---

##  **Performance**

### **Benchmarks**
- **Processing Speed**: 50-200 texts/second (CPU), 500+ texts/second (GPU)
- **Memory Efficiency**: ~2-5MB per 1000 texts (including embeddings)
- **Supported Scale**: 10K+ documents with real-time processing
- **Search Latency**: <100ms for semantic similarity queries

### **Model Performance**
| Model | Embedding Dim | Speed | Quality | Use Case |
|-------|---------------|-------|---------|----------|
| GTE-ModernBERT-Base | 768 | Fast | Excellent | Primary (recommended) |
| all-MiniLM-L6-v2 | 384 | Very Fast | Good | Fallback/lightweight |

---

##  **Contributing**

We welcome contributions! Here's how you can help:

### **Development Setup**
```bash
# Fork the repository
git clone https://github.com/yourusername/text-similarity-explorer.git

# Create feature branch
git checkout -b feature/amazing-feature

# Make your changes and test
python -m pytest tests/

# Submit a pull request
```

### **Areas for Contribution**
-  **New Features** - Additional NLP capabilities or visualizations
-  **Bug Fixes** - Issue resolution and stability improvements
-  **Documentation** - Tutorials, examples, and API docs
-  **UI/UX** - Interface improvements and user experience
-  **Performance** - Optimization and efficiency improvements

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## **Acknowledgments**

### **Built With**
- **[Sentence Transformers](https://www.sbert.net/)** - For state-of-the-art text embeddings
- **[Gradio](https://gradio.app/)** - For the interactive web interface
- **[UMAP](https://umap-learn.readthedocs.io/)** - For dimensionality reduction and visualization
- **[Alibaba NLP](https://huggingface.co/Alibaba-NLP)** - For the GTE-ModernBERT model

### **Inspiration**
This project was inspired by the need for accessible, powerful text analysis tools that combine the latest advances in transformer models with intuitive user interfaces.





<div align="center">

**Made with ❤️ for the NLP and AI community**

*Empowering text analysis through advanced semantic understanding*

[ Star this repo](https://github.com/ayaan-cs/EmbedVista) • [ Report Bug](https://github.com/ayaan-cs/EmbedVista) • [ Request Feature](https://github.com/ayaan-cs/EmbedVista)

</div>