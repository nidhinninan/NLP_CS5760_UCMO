 # CS5760 Natural Language Processing - Homework 2
 
 **Student Name:** Nidhin Ninan
 
 **Student UIN:** 700772413
 
 **Course:** CS 5760, Fall 2025
 
 **Date:** Sept 28, 2025
 
 **HW2 Github Link:** [HW2 Github Link](https://github.com/nidhinninan/NLP_CS5760_UCMO/tree/main/HW2)
 
 **Course Public Repo Link:** [CS 5760 (Nidhin Ninan) Repo Link](https://github.com/nidhinninan/NLP_CS5760_UCMO)
 
 ---
 
 ## Project Overview
 
 This project provides solutions for the second homework assignment in the CS 5760 Natural Language Processing course. The work, implemented in the `HW2_jupyter.ipynb` Jupyter Notebook, covers two key topics in NLP:
 
 1.  **Classification Metrics:** Calculating precision and recall from a confusion matrix, including per-class, macro-averaged, and micro-averaged scores.
 2.  **N-gram Language Modeling:** Implementing a bigram language model from scratch to calculate sentence probabilities and understand model preferences.
 
 ## Solutions Summary
 
 This section details the implementation and key findings for each question.
 
 ### Question 5.3: Confusion Matrix Calculations
 
 **Task Description:**
 The objective was to calculate precision and recall for a multi-class classification problem using a given confusion matrix. The calculations included per-class metrics for "Cat," "Dog," and "Rabbit," as well as macro-averaged and micro-averaged versions of these metrics.
 
 **Implementation Logic:**
 The solution was implemented in Python using the `numpy` library to handle matrix operations.
 1.  The confusion matrix was defined as a `numpy` array.
 2.  For **per-class metrics**, the code iterates through each class to calculate True Positives (TP), False Positives (FP), and False Negatives (FN).
     -   `TP` is the value on the diagonal for that class.
     -   `FP` is the sum of the row for that class, minus the TP.
     -   `FN` is the sum of the column for that class, minus the TP.
 3.  For **macro-averaged metrics**, the per-class precision and recall scores were calculated and then averaged.
 4.  For **micro-averaged metrics**, the total TP, FP, and FN values were aggregated across all classes before calculating a single precision and recall score. This is equivalent to overall accuracy.
 
 The final results are as follows:
 
 *   **Per-class Metrics:**
     -   **Cat:** Precision = 0.2500, Recall = 0.2500
     -   **Dog:** Precision = 0.4444, Recall = 0.4444
     -   **Rabbit:** Precision = 0.4000, Recall = 0.4000
 *   **Averaged Metrics:**
     -   **Macro-averaged Precision:** 0.3648
     -   **Macro-averaged Recall:** 0.3648
     -   **Micro-averaged Precision:** 0.3889
     -   **Micro-averaged Recall:** 0.3889
 
 ### Question 8: Bigram Language Model
 
 **Task Description:**
 This task involved building a bigram language model from a small training corpus. The goal was to compute unigram and bigram counts, estimate bigram probabilities using Maximum Likelihood Estimation (MLE), and use the model to determine which of two given sentences is more probable.
 
 **Implementation Logic:**
 The model was built using Python's `nltk` and `collections` libraries.
 1.  The training corpus was tokenized into sentences and then into words.
 2.  `collections.Counter` was used to compute the frequency of all unigrams and bigrams across the entire corpus.
 3.  Bigram probabilities were calculated using the MLE formula: `P(w_i | w_{i-1}) = Count(w_{i-1}, w_i) / Count(w_{i-1})`. These were stored in a dictionary.
 4.  A function was created to calculate the probability of a full sentence by chaining the probabilities of its constituent bigrams (i.e., multiplying them together).
 5.  The function was used to calculate the probabilities for "<\s> I love NLP <\s>" and "<\s> I love deep learning <\s>".
 
 The model's calculated probabilities were:
 
 *   **P('<\s> I love NLP <\s>')**: 0.3333
 *   **P('<\s> I love deep learning <\s>')**: 0.1667
 
 Based on these results, the model prefers the sentence **'<\s> I love NLP <\s>'** because it has a higher calculated probability.
 
 ---
 
 ### Environment and Dependencies
 
 The notebook was developed and tested using **Python 3.12.5**. The following packages are required to run the code.
 
 ```
 numpy
 nltk
 ```
 
 ### How to Run
 
 1.  Clone the repository to your local machine.
 2.  Set up a Python environment and install the dependencies (e.g., via `pip install numpy nltk`).
 3.  Open `HW2_jupyter.ipynb` in a Jupyter Lab or Jupyter Notebook environment.
 4.  Run all cells sequentially from top to bottom to reproduce the results.
