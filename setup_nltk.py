import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download all required NLTK data
nltk_downloads = [
    'punkt',
    'stopwords',
    'averaged_perceptron_tagger',
    'maxent_ne_chunker',
    'words',
    'wordnet',
    'punkt_tab'
]

print("Downloading NLTK data...")
for item in nltk_downloads:
    try:
        nltk.download(item, quiet=False)
        print(f"✅ Downloaded {item}")
    except Exception as e:
        print(f"❌ Failed to download {item}: {e}")

print("NLTK setup complete!")