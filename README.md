# NIPSNavigtor
## Papers Semantic Search and Summarization System

An NLP-powered platform designed to help researchers quickly locate, understand, and analyze research papers from the NIPS conference. Leveraging advanced natural language processing techniques, this system provides semantic search, concise summarization, and diverse keyword extraction—all in one interactive interface.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Module Overview](#module-overview)
- [Customization](#customization)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## Overview

The **NIPS Papers Semantic Search and Summarization System** is engineered to simplify the research process by transforming complex academic texts into accessible insights. The platform allows users to input a research query and instantly retrieve:

- **Relevant NIPS papers** through semantic search
- **Concise summaries** to quickly grasp core ideas
- **Diverse keywords** that highlight key topics and concepts

By streamlining the navigation of extensive machine learning literature, this system enhances research efficiency and supports informed decision-making.

---

## Features

- **Data Preprocessing:**  
  Cleans raw text by removing HTML tags and unwanted sections. Utilizes spaCy for tokenization, lemmatization, and stop-word removal to ensure high-quality text normalization.

- **Keyword Extraction:**  
  Employs KeyBERT along with SentenceTransformer to extract contextually relevant keywords. Uses Agglomerative Clustering to group similar keywords, ensuring a diverse set of key phrases that capture the paper's main topics.

- **Semantic Search:**  
  Implements a custom Word2Vec model to generate document embeddings, enabling semantic comparisons between user queries and research papers via cosine similarity.

- **Summarization:**  
  Generates concise and coherent summaries of research papers, allowing users to quickly understand the core ideas without reading full texts.

- **Interactive Web Interface:**  
  Built with Streamlit, the application offers an intuitive interface for data upload, processing, and semantic searching, complete with caching and progress spinners for an enhanced user experience.

---

## Installation 

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/JaySGoenka/NIPSNavigator.git

   cd NIPSNavigator

2. **Create a virtual environment:**

  ```bash
   python -m venv venv

   source venv/bin/activate

3. **Install Dependencies:**
  ```bash
   pip install -r requirements.txt

4. **Download the spaCy Language Model:**
  ```bash
  python -m spacy download en_core_web_sm

## Usage

1. **Run the application:**
  ```bash
  streamlit run src/streamlit_app.py

2. **Using the app:**

  - Enter a research query in the search bar.
  - The app will fetch the most relevant NIPS papers based on semantic similarity.
  - It will display concise summaries and extracted key concepts for quick insights.




