import streamlit as st
from conversational_agent import MultiClassConversationalAgent, vectorize_query, generate_answer_with_citations
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the multi-class agent
multi_agent = MultiClassConversationalAgent()

# Read data from files
data_files = ['lecture_notes/intro.txt', 'lecture_notes/capabilities.txt', 'lecture_notes/Harms I.txt', 'lecture_notes/Harms II.txt']
text_chunks = []
for file_name in data_files:
    with open(file_name, 'r', encoding='utf-8') as file:
        text_chunks.extend([line.strip() for line in file.readlines() if line.strip()])

# Add lecture data to the agent
multi_agent.add_class_data('CS324', text_chunks, model)

# Define the Streamlit app
def main():
    st.title('Multi-Class Conversational Agent')

    query = st.text_input('Enter your question:')
    if st.button('Ask'):
        query_vector = vectorize_query(query, model)
        distances, indices = multi_agent.search_class('CS324', query_vector, k=5)
        retrieved_chunks = multi_agent.retrieve_text_chunks('CS324', indices)
        generated_answer = generate_answer_with_citations(retrieved_chunks)

        st.write('Generated Answer:')
        st.write(generated_answer)

if __name__ == '__main__':
    main()
