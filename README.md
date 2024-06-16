# MYFIRSTLLM

This readme file contains project details and information about the MY FIRST LLM project. 

## What is an llm(large language model) ?
An LLM, or Large Language Model, refers to sophisticated AI systems designed to understand and generate human-like text. 
These models are trained on vast amounts of data and can handle complex language tasks, such as translation, summarization, question answering, and text generation.

## Project Description

This project explores the development of a  multi-class conversational agent capable of comprehending and responding to user queries based on preprocessed textual data extracted from diverse lecture notes.
The system leverages advanced Natural Language Processing (NLP) techniques, specifically employing the SentenceTransformer model for text vectorization and FAISS (Facebook AI Similarity Search) for efficient 
retrieval of relevant information.


## Data Collection
The lecture notes were gathered from the Stanford LLMs at https://stanford-cs324.github.io/winter2022/lectures/.

## Project directory detail 
MYFIRSTLLMS/
├── _pycache_/
├── lecture_notes/
│   ├── ≡ capabilities.txt
│   ├── Harms I.txt
│   ├── Harms II.txt
│   ├── intro.txt
│   └── table.txt
├── app.log
├── dataprocesing.ipynb
├── ≡ format.txt
├── lecture_notes.db
├── lecture_notes.index
├── lecture_proc_streamlit.py
├── lecture_processing.py
└── LLMnotebook.ipynb

