# 🔮 SmoothLM: N-gram Language Modeling & Smoothing

![Language Models](https://img.shields.io/badge/NLP-Language%20Models-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-green)
![Assignment](https://img.shields.io/badge/IIIT%20Hyderabad-CS7.401%20iNLP-orange)

> A comprehensive implementation of n-gram language models with various smoothing techniques for natural text generation and analysis.

## 📋 Overview

SmoothLM is an implementation of n-gram language models with three different smoothing techniques (Laplace, Good-Turing, and Linear Interpolation). This project includes tools for text tokenization, language model training, next-word prediction, and perplexity calculation—providing a complete toolkit for understanding and implementing statistical language models.

## ✨ Features

- 📚 **Tokenization**: Custom regex-based tokenizer supporting various text patterns
- 🧩 **N-gram Models**: Implementation of n-gram models (n=1,3,5)
- 🧠 **Smoothing Techniques**:
  - Laplace (Add-One) Smoothing
  - Good-Turing Smoothing
  - Linear Interpolation
- 🔍 **Text Generation**: Next word prediction using trained models
- 📊 **Perplexity Analysis**: Calculate and visualize perplexity scores
- 🧪 **OOD Testing**: Evaluation of model behavior in out-of-distribution scenarios

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SmoothLM.git
cd SmoothLM

# Set up a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🛠️ Usage

### Tokenization

```bash
python3 tokenizer.py
# Input prompt will appear
# Example: "Is that what you mean? I am unsure."
# Output: [['Is', 'that', 'what', 'you', 'mean', '?'], ['I', 'am', 'unsure', '.']]
```

### Language Model Training & Evaluation

```bash
python3 language_model.py <lm_type> <corpus_path>
# lm_type: 'l' (Laplace), 'g' (Good-Turing), 'i' (Interpolation)
# corpus_path: path to training corpus

# Example:
python3 language_model.py i ./corpus/pride_and_prejudice.txt
# Input prompt: "I am a woman."
# Output: score: 0.69092021
```

### Text Generation

```bash
python3 generator.py <lm_type> <corpus_path> <k>
# lm_type: 'l' (Laplace), 'g' (Good-Turing), 'i' (Interpolation)
# corpus_path: path to training corpus
# k: number of candidate words to display

# Example:
python3 generator.py i ./corpus/pride_and_prejudice.txt 3
# Input: "An apple a day keeps the doctor"
# Output:
# away 0.4
# happy 0.2
# fresh 0.1
```

### Perplexity Calculation

```bash
python3 perplexity.py <lm_type> <corpus_path> <n> <split>
# lm_type: 'l' (Laplace), 'g' (Good-Turing), 'i' (Interpolation)
# corpus_path: path to corpus
# n: n-gram size (1, 3, or 5)
# split: 'train' or 'test'

# Example:
python3 perplexity.py g ./corpus/ulysses.txt 3 test
# Outputs perplexity scores to a file
```

## 📊 Results & Analysis

### Generation Results

The project evaluates different n-gram models and smoothing techniques for text generation:

#### No Smoothing
- **Unigram (N=1)**: Poor performance with incoherent and random predictions
- **Trigram (N=3)**: Improved but still lacks fluency, with subject/meaning inconsistencies
- **5-gram (N=5)**: Best performance among non-smoothed models, most coherent results

#### With Smoothing
- **Laplace**: Improved handling of unseen contexts, but overall weaker than other methods
- **Good-Turing**: Best overall performance, especially for higher n-values
- **Linear Interpolation**: Good performance that could be improved with better λ weights

### Perplexity Analysis

Key findings from perplexity evaluation:
- Test set perplexity consistently higher than training set perplexity
- Good-Turing outperforms Laplace for lower-order n-grams
- Linear interpolation shows challenges with data sparsity in higher-order n-grams
- Laplace smoothing produces higher perplexity than other methods

## 📂 Project Structure

```
SmoothLM/
├── tokenizer.py                            # Text tokenization implementation
├── language_model.py                       # Language model implementation
├── generator.py                            # Text generation implementation
├── script.py                               # Perplexity calculation
├── graph.py                                # Plotting graphs
├── Pride and Prejudice - Jane Austen.txt   # Jane Austen's novel corpus
├── Ulysses - James Joyce.txt               # James Joyce's novel corpus
├── output/
│   ├── 2022101094_good_turing_1_test_perplexity_pride            # Perplexity score files
     .....     
└── README.md               
```

## 🔍 Implementation Details

### Tokenization

The tokenizer handles various text patterns including:
- Regular words and punctuation
- URLs, hashtags, and mentions
- Percentages and numerical expressions
- Time expressions and periods

### Good-Turing Smoothing

Implementation follows the formula:

P<sub>GT</sub>(w<sub>1</sub>...w<sub>n</sub>) = r*/N

where r* = (r+1)S(r+1)/S(r)

For unseen events: P<sub>GT</sub>(w<sub>1</sub>...w<sub>n</sub>) = N<sub>1</sub>/N

### Linear Interpolation

Combines multiple n-gram models with λ weights:

P(w<sub>n</sub>|w<sub>1</sub>...w<sub>n-1</sub>) = λ<sub>1</sub>P<sub>1</sub>(w<sub>n</sub>) + λ<sub>2</sub>P<sub>2</sub>(w<sub>n</sub>|w<sub>n-1</sub>) + ... + λ<sub>n</sub>P<sub>n</sub>(w<sub>n</sub>|w<sub>1</sub>...w<sub>n-1</sub>)

## 📝 Assignment Details

This project was developed as part of the Introduction to NLP (CS7.401) course at IIIT Hyderabad for Spring 2025. The implementation follows the assignment guidelines and was completed by January 23rd, 2025.

## 🔗 References

1. [Introduction to N-Gram Language Modeling Methods](https://towardsdatascience.com/introduction-to-language-models-n-gram-e323081503d9)
2. [Jurafsky & Martin - Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)
3. [Good-Turing Smoothing Paper](https://www.cs.toronto.edu/~frank/csc401/readings/Good1953.pdf)
4. [Linear Interpolation for Language Modeling](https://www.cs.cornell.edu/courses/cs4740/2014sp/lectures/smoothing+backoff.pdf)


## 👤 Author

[Mayank Mittal] - 2022101094
International Institute of Information Technology, Hyderabad
