# CS5760 Natural Language Processing - Homework 4

**Student Name:** Nidhin Ninan

**Student UIN:** 700772413

**Course:** CS 5760, Fall 2025

**Date:** November 18, 2025

**HW4 Github Link:** [HW4 Github Link](https://github.com/nidhinninan/NLP_CS5760_UCMO/tree/main/HW4)

**Course Public Repo Link:** [CS 5760 (Nidhin Ninan) Repo Link](https://github.com/nidhinninan/NLP_CS5760_UCMO)

---

## Project Overview

This project provides solutions for the fourth homework assignment in the CS 5760 Natural Language Processing course. The work is implemented in three separate Jupyter Notebooks, each addressing a key concept in modern NLP:

1.  **Q1_Character_Level_RNN.ipynb:** Implements a character-level Recurrent Neural Network (RNN) to predict the next character in a sequence.
2.  **Q2_Mini_Transformer_Encoder.ipynb:** Builds a mini-Transformer Encoder from scratch, including multi-head self-attention and positional encoding.
3.  **Q3_Scaled_Dot_Product_Attention.ipynb:** Implements the scaled dot-product attention mechanism, a core component of Transformers.

## Solutions Summary

This section details the implementation and key findings for each programming question.

### Question 1: Character-Level RNN Language Model

**Task Description:**
The goal was to train a tiny character-level RNN to predict the next character given a sequence of previous characters. The model was first trained on a small toy corpus and then designed to be expandable to a larger text file. The implementation includes an embedding layer, an RNN (LSTM) layer, and a linear output layer, trained with teacher forcing.

**Implementation Logic:**
The solution is implemented in `Q1_Character_Level_RNN.ipynb`.
1.  **Data Preparation:** A small toy corpus is created. A character vocabulary is built, and mappings from characters to indices (and vice-versa) are established. The corpus is converted into a sequence of numerical indices.
2.  **Sequence Creation:** The indexed data is transformed into input-target pairs of sequences for training.
3.  **Model Definition:** A `CharRNN` class is defined using PyTorch, consisting of an `nn.Embedding` layer, an `nn.LSTM` layer, and an `nn.Linear` layer to map the RNN output to the vocabulary size.
4.  **Training:** The model is trained using the Adam optimizer and Cross-Entropy Loss. Teacher forcing is used, where the true previous character is fed as input at each step during training.
5.  **Text Generation:** A function `generate_text` is implemented to produce new text by sampling from the model's output distribution. Temperature-controlled sampling is used to control the creativity of the generated text.
6.  **Analysis:** The training loss is plotted, and the effect of varying sequence length, hidden size, and temperature is discussed.

**Code:**
The full implementation and code can be found in the [Q1_Character_Level_RNN.ipynb](Q1_Character_Level_RNN.ipynb) file.

### Question 2: Mini Transformer Encoder for Sentences

**Task Description:**
The task was to build a mini-Transformer Encoder to process a batch of sentences. This involved implementing several key components of the Transformer architecture: sinusoidal positional encoding, multi-head self-attention, feed-forward layers, and residual connections with layer normalization.

**Implementation Logic:**
The model is built in `Q2_Mini_Transformer_Encoder.ipynb`.
1.  **Data and Tokenization:** A small dataset of 10 sentences is created. A vocabulary is built, and sentences are tokenized and padded to a fixed length.
2.  **Positional Encoding:** A sinusoidal positional encoding function is implemented to inject information about the position of tokens in the sequence.
3.  **Attention Mechanism:** A `MultiHeadAttention` module is created, which internally uses the `scaled_dot_product_attention` function. This module splits the model's dimension into multiple heads, allowing it to jointly attend to information from different representation subspaces.
4.  **Encoder Layer:** A `TransformerEncoderLayer` is defined, which combines the multi-head attention sub-layer and a position-wise feed-forward network. Residual connections and layer normalization (`AddNorm`) are applied after each sub-layer.
5.  **Full Encoder:** The final `TransformerEncoder` stacks multiple `TransformerEncoderLayer` modules. It includes the initial embedding layer and adds the positional encodings to the token embeddings.
6.  **Visualization:** The model processes the batch of sentences, and the attention weights from one of the heads are visualized as a heatmap to show how words in a sentence attend to each other.

**Code:**
The full implementation and code can be found in the [Q2_Mini_Transformer_Encoder.ipynb](Q2_Mini_Transformer_Encoder.ipynb) file.

### Question 3: Implement Scaled Dot-Product Attention

**Task Description:**
The objective was to implement the scaled dot-product attention function from scratch. The implementation was tested with random Q, K, and V inputs, and a key part of the task was to demonstrate and explain the importance of the scaling factor (1/√dₖ).

**Implementation Logic:**
The function is implemented in `Q3_Scaled_Dot_Product_Attention.ipynb`.
1.  **Function Definition:** A Python function `scaled_dot_product_attention` is created that takes Q, K, and V tensors as input.
2.  **Score Calculation:** It computes the dot product of Q and Kᵀ.
3.  **Scaling:** The resulting scores are divided by the square root of the key dimension (dₖ).
4.  **Softmax:** A softmax function is applied to the scaled scores to obtain the attention weights.
5.  **Output Calculation:** The attention weights are multiplied by the V tensor to produce the final output.
6.  **Stability Check:** The notebook includes a detailed check to show the effect of the scaling factor. It computes the softmax output on scores both *before* and *after* scaling. The results demonstrate that without scaling, large dot product values can push the softmax function into regions with very small gradients, making training unstable. Scaling keeps the variance of the scores in a reasonable range, leading to a "softer" attention distribution and more stable training.

**Code:**
The full implementation and code can be found in the [Q3_Scaled_Dot_Product_Attention.ipynb](Q3_Scaled_Dot_Product_Attention.ipynb) file.

---

### Environment and Dependencies

The notebooks were developed and tested using **Python 3.11**. The following packages are required to run the code.

```
torch
numpy
matplotlib
seaborn
```

You can install them via pip:
```bash
pip install torch numpy matplotlib seaborn
```

### How to Run

1.  Clone the repository to your local machine.
2.  Set up a Python environment and install the dependencies listed above.
3.  Open the Jupyter Notebooks (`Q1_Character_Level_RNN.ipynb`, `Q2_Mini_Transformer_Encoder.ipynb`, `Q3_Scaled_Dot_Product_Attention.ipynb`) in a Jupyter Lab or Jupyter Notebook environment.
4.  Run all cells sequentially from top to bottom in each notebook to reproduce the results.
