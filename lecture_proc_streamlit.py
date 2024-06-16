pip install nltk
import streamlit as st
import os
import logging
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Set environment variables to suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'


logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

#preprocessing text
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

#vectorizing text chunks
def vectorize_text_chunks(text_chunks, model):
    vectors = [model.encode(chunk) for chunk in text_chunks]
    return np.array(vectors)

# query vectorization
def vectorize_query(query, model):
    return model.encode([query])


def generate_answer_with_citations(retrieved_chunks):
    answer = ""
    for chunk in retrieved_chunks:
        answer += chunk + " "
    return answer.strip()

# Multi-class conversational agent
class MultiClassConversationalAgent1:
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

# model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Read data 
data_files = ['lecture_notes\intro.txt', 'lecture_notes\capabilities.txt', 'lecture_notes\Harms I.txt', 'lecture_notes\Harms II.txt']
text_chunks = []
for file_name in data_files:
    with open(file_name, 'r', encoding='utf-8') as file:
        text_chunks.extend([line.strip() for line in file.readlines() if line.strip()])

# multi-class agent
multi_agent = MultiClassConversationalAgent1()


multi_agent.add_class_data('CS324', text_chunks, model)

# Streamlit 
st.title('MY FIRST LLM (Large language model)')

# queries Input
query = st.text_input('Enter your query here:')
if query:

    query_vector = vectorize_query(query, model)

    distances, indices = multi_agent.search_class('CS324', query_vector, k=5)

    retrieved_chunks = multi_agent.retrieve_text_chunks('CS324', indices)

    generated_answer = generate_answer_with_citations(retrieved_chunks)

    st.subheader('Generated Answer:')
    st.write(generated_answer)
