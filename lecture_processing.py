import os
import logging
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Set environment variables to suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Preprocess text
def preprocess_text(text):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text

# Vectorize text chunks
def vectorize_text_chunks(text_chunks, model):
    vectors = [model.encode(chunk) for chunk in text_chunks]
    return np.array(vectors)

# Vectorize query
def vectorize_query(query, model):
    return model.encode([query])

# Generate answer with citations
def generate_answer_with_citations(retrieved_chunks):
    answer = " ".join(retrieved_chunks)
    return answer.strip()

# Multi-class conversational agent
class MultiClassConversationalAgent:
    def __init__(self):
        self.indices = {}
        self.text_chunks = {}

    def add_class_data(self, class_name, text_chunks, model):
        vectors = vectorize_text_chunks(text_chunks, model)
        dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        self.indices[class_name] = index
        self.text_chunks[class_name] = text_chunks

    def search_class(self, class_name, query_vector, k=5):
        try:
            index = self.indices[class_name]
            distances, indices = index.search(query_vector, k)
            return distances, indices
        except KeyError as e:
            logging.error(f"Class {class_name} not found: {e}")
            return None, None
        except Exception as e:
            logging.error(f"Error in search_class: {e}")
            return None, None

    def retrieve_text_chunks(self, class_name, indices):
        try:
            text_chunks = self.text_chunks[class_name]
            retrieved_chunks = [text_chunks[idx] for idx in indices[0]]
            return retrieved_chunks
        except KeyError as e:
            logging.error(f"Class {class_name} not found: {e}")
            return []
        except IndexError as e:
            logging.error(f"Index error: {e}")
            return []
        except Exception as e:
            logging.error(f"Error in retrieve_text_chunks: {e}")
            return []

if __name__ == "__main__":
    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Read data from files
    data_files = ['lecture_notes/intro.txt', 'lecture_notes/capabilities.txt', 'lecture_notes/Harms I.txt', 'lecture_notes/Harms II.txt']
    text_chunks = []
    for file_name in data_files:
        with open(file_name, 'r', encoding='utf-8') as file:
            text_chunks.extend([line.strip() for line in file.readlines() if line.strip()])

    # Initialize the multi-class agent
    multi_agent = MultiClassConversationalAgent()

    # Add lecture data to the agent
    multi_agent.add_class_data('CS324', text_chunks, model)

    # Define queries
    queries = [
        "What are some milestone model architectures and papers in the last few years?",
        "What is Penn Tree Bank?",
        "Can large language models answer questions?"
    ]

    for query in queries:
        query_vector = vectorize_query(query, model)
        distances, indices = multi_agent.search_class('CS324', query_vector, k=5)
        retrieved_chunks = multi_agent.retrieve_text_chunks('CS324', indices)
        generated_answer = generate_answer_with_citations(retrieved_chunks)

        print(f"Query: {query}")
        print("Generated Answer:")
        print(generated_answer)
        print("\n")
