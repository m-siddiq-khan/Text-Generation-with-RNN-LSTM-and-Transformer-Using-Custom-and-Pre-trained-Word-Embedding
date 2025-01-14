# Text-Generation-with-RNN-LSTM-and-Transformer-Using-Custom-and-Pre-trained-Word-Embedding
### Overview
This project aims to train and evaluate four different text generation models, leveraging both RNN (LSTM) and Transformer architectures, with two variations each: one using custom-made embeddings and the other utilizing pre-trained embeddings. The data for this project was scraped from three different websites, providing a diverse set of text sources for training. The primary objective is to compare the performance of these models in text generation tasks, focusing on the quality of generated text using BLEU and ROUGE scores. The project involves building and training each model separately, processing and preparing the text data, and analyzing the generated output. The results highlight the performance of each model variation and demonstrate the impact of embedding choice on model effectiveness.

### Features
<span style="font-size:24px;">•</span> **Model Variations**:

The project trains four distinct models, each evaluating different architecture and embedding approaches:

**RNN with Custom Embeddings**: Uses embeddings generated from scratch.

**RNN with Pre-trained Embeddings**: Utilizes pre-trained embeddings (Sentence Transformer).

**Transformer with Custom Embeddings**: Leverages custom embeddings for a Transformer-based model.

**Transformer with Pre-trained Embeddings**: Uses pre-trained embeddings (Sentence Transformer) in a Transformer architecture.

### Data Preprocessing

The dataset consists of text data from multiple sources. Key preprocessing steps include:

**Text Merging**: Combining data from various group members into a single dataset.

**Punctuation Removal**: Cleaning text to remove unwanted punctuation.

**Tokenization**: Breaking down the text into tokens and building a vocabulary of size 46,372.


### Model Architecture

**RNN Models**:
Use of LSTM layers in both RNN models to capture long-term dependencies, with 512 units in each LSTM layer.

Dropout layers are added after each LSTM layer to reduce overfitting (rate of 0.2).

**Transformer Models**:

Uses Positional Encoding to inject information about word positions in the sequence.

The Decoder Layer is used to transform the embeddings into the output sequence, with 8 layers for the Transformer model.

Each model is trained with Adam optimizer and MSE loss for RNN models and Cross-Entropy loss for Transformer models.


### Performance Evaluation

The performance of each model is evaluated using the following metrics:

**BLEU Score**: Measures the quality of generated text based on n-gram precision.

**ROUGE Scores**: Evaluates the overlap between the generated and reference text (ROUGE-1, ROUGE-L).

Example results from the models are:

**RNN with Custom Embeddings**: BLEU score: 0.01, ROUGE-1 F1: 0.200, ROUGE-L F1: 0.1471.

**RNN with Pre-trained Embeddings**: BLEU score: 0.11, ROUGE-1 F1: 0.1250, ROUGE-L F1: 0.200.

**Transformer with Custom Embeddings**: BLEU score: 0.1494, ROUGE-1 F1: 0.4838, ROUGE-L F1: 0.4159.

**Transformer with Pre-trained Embeddings**: BLEU score: 0.8417, ROUGE-1 F1: 0.9108, ROUGE-L F1: 0.9056.
<span style="font-size:24px;">•</span> Challenges Faced:

### Technologies Used

<span style="font-size:24px;">•</span> **Python**: The primary programming language used for data preprocessing, model building, and training.

<span style="font-size:24px;">•</span> **TensorFlow/Keras and PYTorch**: The framework used for building, compiling, and training RNN (LSTM) and Transformer models.

<span style="font-size:24px;">•</span> **Pre-trained Embedding Models**:

**RNN Model**: Sentence Transformer (384 dimensions) for embedding extraction

**Transformer Model**: FastText embeddings were used for input representation.

<span style="font-size:24px;">•</span> **Data Scraping Tools**: BeautifulSoup for scraping text data from websites, along with necessary libraries such as requests and lxml for fetching and parsing web content.

<span style="font-size:24px;">•</span> **Pre-trained Embedding Models**: Sentence Transformer and custom embeddings.

<span style="font-size:24px;">•</span> **Evaluation Metrics**: BLEU and ROUGE scores for model performance analysis
