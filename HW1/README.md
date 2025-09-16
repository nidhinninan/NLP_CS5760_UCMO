# CS5760 Natural Language Processing - Homework 1

**Student Name:** Nidhin Ninan

**Student UIN:** 700772413

**Course:** CS 5760, Fall 2025

**Date:** Sept 15, 2025

**Github Link:** [HW1 Github Link](https://github.com/nidhinninan/NLP_CS5760_UCMO/tree/main/HW1)

**Course Public Repo Link:** [CS 5760 (Nidhin Ninan) Repo Link](https://github.com/nidhinninan/NLP_CS5760_UCMO)

---

## Project Overview

This project provides solutions for the first homework assignment in the CS 5760 Natural Language Processing course. The work, implemented in the `HW1.ipynb` Jupyter Notebook, covers four fundamental topics in NLP:

1.  **Regular Expressions:** Crafting precise regex patterns to capture various linguistic and numerical formats.
2.  **Text Tokenization:** Implementing and comparing different tokenization strategies for Malayalam, a morphologically rich language.
3.  **Byte Pair Encoding (BPE):** Manually and programmatically implementing the BPE algorithm to learn subword tokenization.
4.  **Edit Distance:** Calculating the Levenshtein distance between two strings using dynamic programming with different cost models.

## Solutions Summary

This section details the implementation and key findings for each question.

### Question 1: Regular Expressions

**Task Description:**
The objective was to create six distinct regular expression patterns to match specific text formats, including U.S. ZIP codes, words not starting with a capital letter, complex numbers, spelling variants of "email," interjections, and lines ending with a question mark.

**Implementation Logic:**
Each pattern was constructed using specific regex tokens to meet the requirements. For example, `\b` was used for whole-word boundaries, `(?:...)` for non-capturing groups, and lookarounds like `(?<!\S)` to ensure matches were not embedded within other tokens.

The final patterns are as follows:

*   **U.S. ZIP Codes:**
    ```regex
    \b\d{5}(?:[-\s]\d{4})?\b
    ```
*   **Words NOT Starting with a Capital Letter (ASCII):**
    ```regex
    \b(?![A-Z])[a-z][a-z\'\-]*\b
    ```
*   **Comprehensive Numbers:**
    ```regex
    (?<!\S)[+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?(?! \S)
    ```
*   **"Email" Variants (Case-Insensitive):**
    ```regex
    \b[eE](?:[-– ])?[mM][aA][iI][lL]\b
    ```
*   **"Go" Interjections:**
    ```regex
    (?<!\S)go+[!.,?]?(?!\S)
    ```
*   **Lines Ending with "?":**
    ```regex
    \?[\t\"')\]\}\"\'»]*$
    ```

### Question 2: Tokenization

**Task Description:**
This task involved implementing and comparing three different tokenization methods on a Malayalam paragraph: a naïve space-based splitter, a manual rule-based tokenizer, and a pre-trained tool-based tokenizer (`google/muril-base-cased`). The goal was to analyze their effectiveness on a morphologically complex language.

**Implementation Logic:**
1.  **Naïve Tokenizer:** Used Python's simple `text.split()` method.
2.  **Manual Tokenizer:** Implemented a function with hand-crafted rules to separate punctuation. Although rules for English clitics were included per the assignment, it was noted that they do not apply to Malayalam.
3.  **Tool-based Tokenizer:** Utilized the `AutoTokenizer` from the `transformers` library with the `google/muril-base-cased` model, which performs WordPiece subword tokenization.


**Key Reflections:**

**_Observation based on the above tokenisation of sentences in Malayalam:_**

1.  **WordPiece vs. Rule-based Tokenization:** The google/muril-base-cased model using WordPiece tokenization handles the Malayalam paragraph much more effectively than naive or manual approaches, breaking complex Malayalam words into meaningful subword units marked with ## continuation symbols that preserve morphological information.

2.  **Malayalam Script Complexity:** Malayalam's agglutinative nature and complex script with conjuncts, vowel signs, and compound words pose significant challenges for simple tokenization methods, but WordPiece tokenization trained on multilingual data can capture these linguistic patterns more naturally.

3.  **Subword Granularity Benefits:** The ## prefix system in WordPiece allows the model to represent long Malayalam words (like "പരീക്ഷിച്ചുകൊണ്ടിരിക്കുകയാണ്") as sequences of meaningful subword pieces, enabling better handling of unseen words and morphological variations compared to whole-word approaches.

4.  **Cross-script Robustness:** When processing mixed Malayalam-English text in our paragraph, the pre-trained google/muril model seamlessly handles both scripts without requiring separate tokenization rules, demonstrating the advantage of multilingual models over language-specific manual rules.

5.  **Trade-offs in Semantic Preservation:** While WordPiece tokenization excels at morphological decomposition and OOV handling, it may fragment semantically coherent units like proper names ("ഡോ. കെ.എം. നായരിന്") or compound expressions, requiring careful consideration for downstream tasks that depend on preserving semantic boundaries.

**_Reflection on the Tokenisation based of the solution_**:

The most challenging aspect of tokenizing Malayalam is its agglutinative morphology, where multiple suffixes and particles attach to a root word, creating long, complex forms that are difficult to segment correctly. Compared to English, which has more isolated word structures and clearer space delimiters, Malayalam tokenization must handle intricate word boundaries that aren't always separated by spaces. Punctuation, complex morphology, and multiword expressions (MWEs) significantly increase this difficulty, as they require a deeper linguistic understanding beyond simple rules to preserve the correct meaning of tokens.

### Question 3: Byte Pair Encoding (BPE)

**Task Description:**
The task was to implement the Byte Pair Encoding (BPE) algorithm from scratch. This involved a step-by-step manual demonstration, a full programmatic implementation on a toy English corpus, and finally, training a BPE model on the Malayalam paragraph from Question 2.

**Implementation Logic:**
A `BPELearner` class was created to handle the BPE process. The core logic involves:
1.  Initializing the vocabulary with individual characters and an end-of-word marker (`_`).
2.  **Iterative Merging:** The main `learn_bpe` method iterates for a specified number of merges. In each step:
    *   The `get_pairs` helper function scans the entire corpus vocabulary to count the frequency of all adjacent symbol pairs.
    *   The most frequent pair is identified using `max()` on the pair counts.
    *   The `merge_vocab` function creates a new version of the corpus vocabulary where every occurrence of this most frequent pair is replaced with a new, single merged symbol (e.g., `'e' 'r'` becomes `'er'`).
    *   This new vocabulary becomes the input for the next iteration.
3.  **Storing Merges:** The sequence of merges is stored in order, as this represents the learned tokenization model.

A separate function, `apply_bpe_segmentation`, was implemented to tokenize new words. It takes a word, splits it into characters, and then iteratively applies the learned merges in the exact order they were created to produce the final subword segmentation.


**Key Reflections:**

**_Why use Subword Tokenisation_**

Subword tokenization effectively solves the out-of-vocabulary (OOV) problem by breaking down unknown words into a sequence of known, smaller pieces. For instance, even though the invented word "newestest" was not in the original corpus, the BPE model could still represent it by segmenting it into subwords like `['new', 'e', 's', 't', 'e', 's', 't', '_']`.

This approach ensures that no word is ever truly "unknown," only a new combination of existing vocabulary parts. A perfect example of subwords aligning with meaningful linguistic units, or morphemes, is the creation of the token `er_`. This subword directly corresponds to the English comparative suffix, as seen in the segmentation of "newer" and "wider." By learning this common suffix, the model captures a fundamental piece of English grammar, allowing it to better understand and generalize across different words.

Take the word, "faster", as an example for an unknown word that is not in the BPE corpus. The token `er_` would segment the word by separating the known stem from the suffix, resulting in a tokenization like `['fast', 'er_']`. This process allows the model to correctly interpret "faster" as the comparative form of "fast," even if it had never encountered the specific word "faster" during its training, perfectly illustrating its power to generalize.

**_Observation and Reflections from the mini-BPE that we just learned_**

The BPE algorithm learned a variety of subword types, reflecting a hierarchy from simple character combinations to more linguistically meaningful units.

*   Meaningful Suffixes: The most interesting learned tokens were common Malayalam suffixes, which are true morphemes. For example, `ൾ_` (Merge 6) is a standard plural marker, and `ൽ_` (Merge 15) often represents the locative case (meaning "in" or "at").
*   Whole Words: For very high-frequency words, the algorithm managed to reconstruct the entire word. A great example is `അത്_` (ath_, meaning "it" or "that"), which was formed in Merge 13, and `ഇന്നലെ_` (innaley_, meaning "yesterday") in Merge 22.
*   Common Character Combinations: Many of the initial merges were simply statistically frequent character pairs or consonant clusters, not necessarily meaningful units on their own. Examples include `കക` (kk), `നന` (nn), and `തയ` (thay). These act as basic building blocks for larger tokens.

**NOTE**: *With only 30 merges, the process primarily captured suffixes and very frequent short words. It didn't progress enough to consistently isolate word stems or prefixes.*

**Pros and Cons**

Subword tokenization is a powerful technique for morphologically rich languages like Malayalam, but it comes with trade-offs.

**Pros**

1.  **Handles Agglutinative Morphology:** Malayalam words are often formed by adding multiple suffixes to a root stem. Subword tokenization naturally handles this by breaking words into a stem and its constituent suffixes. This allows a model to recognize the relationship between പോകുന്നു (pokunnu - "is going") and പോയി (poyi - "went") by seeing a common stem token, drastically reducing the vocabulary size and helping it generalize to unseen word forms.
2.  **Manages Compound Words and Loanwords:** The method gracefully handles both native compound words (സമാസപദം) and the many English loanwords in Malayalam like "റിപ്പോർട്ട്" (report). Instead of treating "റിപ്പോർട്ട്" as an unknown out-of-vocabulary (OOV) word, BPE breaks it down into manageable subwords (like `റപപർടട_` in your output), allowing the model to process it effectively.

**Cons**

1.  **Creates Non-Meaningful Splits:** Since BPE is driven purely by frequency, it can create subwords that have no linguistic or semantic meaning. For example, splitting a word based on a common but meaningless character pair can make it harder for the model to learn the true compositional meaning of the word. A split might be statistically optimal but linguistically nonsensical.
2.  **Struggles with "Sandhi":** Malayalam uses complex "Sandhi" rules, where sounds at word junctions merge and change (e.g., അതി + ഇൽ → അതിൽ). A BPE model trained on a small corpus might not learn these phonetic rules and could split a merged word in an unnatural way that obscures the original root words, making it difficult for a model to capture the underlying semantics.

### Question 4: Edit Distance

**Task Description:**
This question required implementing the Levenshtein edit distance algorithm using dynamic programming to find the minimum cost to transform "sunday" into "saturday". The implementation had to support two different cost models and show the full DP table and the optimal edit sequence for each.

**Implementation Logic:**
An `EditDistance` class was implemented with the following features:
*   A `compute_distance` method that builds and fills a dynamic programming (DP) table (a 2D `numpy` array) of size `(len(source)+1) x (len(target)+1)`.
    *   The table is initialized with base cases: the cost of deleting all source characters to match an empty target (first column) and inserting all target characters from an empty source (first row).
    *   It then iterates through the table, filling each cell `dp[i][j]` with the minimum cost to transform the first `i` characters of the source to the first `j` characters of the target. This cost is the minimum of three possibilities: a substitution/match (`dp[i-1][j-1]`), an insertion (`dp[i][j-1]`), or a deletion (`dp[i-1][j]`), plus the corresponding operation cost.
*   A `backtrace` method that reconstructs the optimal edit path by traversing the completed DP table backward from the bottom-right corner to the top-left. At each cell, it determines which of the three possible prior cells (diagonal, left, or top) led to the current cell's minimum cost, thereby identifying the operation (match/substitute, insert, or delete) performed at that step.
*   The class was instantiated twice with different cost parameters to test two models:
    *   **Model A:** Substitution = 1, Insertion = 1, Deletion = 1
    *   **Model B:** Substitution = 2, Insertion = 1, Deletion = 1

**Key Reflections:**

**_Reflection on Edit Distance Model Comparison_**

*   The two models produced different distances
    *   Model B assigns a higher cost of 2 for a substitution, while Model A's cost is only 1.
*   For transforming "Sunday" to "Saturday," the most useful operations were two insertions and one substitution.
*   The choice of model is critical for different applications:
    *   **Spell Check:** For spell-checking tasks where single-character substitutions are common typos, a low substitution cost makes sense. Model A's equal costs are generally better, as common typos like substitutions, insertions, or deletions are treated as equally probable errors.
    *   **DNA Alignment:** In genetics, a substitution (a point mutation) is a fundamentally different biological event than an insertion or deletion (an indel). In DNA sequence alignment, costs should reflect real biological mutation rates. Model B's variable costs are superior, as they can represent the different biological probabilities of a mutation (substitution) versus an indel (insertion/deletion).

---

### Environment and Dependencies

The notebook was developed and tested using **Python 3.12.5**. The following packages are required to run the code, particularly for the tool-based tokenization in Question 2.

```
transformers==4.56.1
torch==2.8.0
numpy==2.1.0
pandas==2.2.2
matplotlib==3.9.2

Tokeniser Pre-Trained Model: google/muril-base-cased model

# PyTorch CUDA Toolkit Dependencies (cu12)
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
```

**Note:** While a CUDA-enabled environment is beneficial for the performance of the `transformers` library, it is not strictly required to run this notebook, as the operations are not computationally intensive.

### How to Run

1.  Clone the repository to your local machine.
2.  Set up a Python environment and install the dependencies listed in the section above (e.g., using `pip install -r requirements.txt`).
3.  Open `HW1.ipynb` in a Jupyter Lab or Jupyter Notebook environment.
4.  Run all cells sequentially from top to bottom to reproduce the results and analyses.
