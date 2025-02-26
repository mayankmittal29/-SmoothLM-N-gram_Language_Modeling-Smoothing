import random
import math
from tokenizer import (
    tokenize_text,
)
from language_model import LaplaceLanguageModel, GoodTuringLanguageModel, InterpolationLanguageModel
import sys

def split_corpus(corpus, test_size=1000):
    random.shuffle(corpus)
    test_corpus = corpus[:test_size]
    train_corpus = corpus[test_size:]
    return train_corpus, test_corpus


def calculate_perplexity(lm, sentence):
    sentence=["<s>"]*(lm.n-1)+sentence+["</s>"]
    log_probability = lm.get_sentence_probability(sentence)
    sentence_length = len(sentence)
    perplexity = math.exp(-log_probability / sentence_length)
    return perplexity


def generate_perplexity_file(lm, corpus, file_path):
    total_perplexity = 0.0
    with open(file_path, "w") as f:
        final_list = []
        for sentence in corpus:
            perplexity = calculate_perplexity(lm, sentence)
            total_perplexity += perplexity
            final_list.append([sentence, perplexity])
        avg_perplexity = total_perplexity / len(corpus)
        f.write(f"Average perplexity: {avg_perplexity}\n")
        for item in final_list:
            f.write(f"{' '.join(item[0])}\t{item[1]}\n")


def main():
    # Ask for the corpus file location
    if len(sys.argv) < 2:
        print("Usage: python3 perplex.py <corpus_path>")
        sys.exit(1)

    corpus_path = sys.argv[1]

    # Load and tokenize the corpus
    with open(corpus_path, "r") as f:
        corpus_content = f.read()
    corpus = tokenize_text(corpus_content)

    # Split the corpus into train and test sets 
    train_corpus, test_corpus = split_corpus(corpus)
    n_values = [1, 3, 5]
    lm_types = {
        "laplace": LaplaceLanguageModel,
        "good_turing": GoodTuringLanguageModel,
        "interpolation": InterpolationLanguageModel,
    }

    if "pride" in corpus_path or "Pride" in corpus_path:
        dataset = "pride"
    elif "Ulysses" in corpus_path or "ulysses" in corpus_path:
        dataset = "ulysses"

    # Generate perplexity files for each combination
    for lm_name, lm_class in lm_types.items():
        for n in n_values:
            lm = lm_class(n=n)

            lm.train(train_corpus)

            # Generate perplexity files for the train set
            train_file_path = f"output/2022101094_{lm_name}_{n}_train_perplexity_{dataset}.txt"
            generate_perplexity_file(lm, train_corpus, train_file_path)
            print(f"Generated {train_file_path}")

            # Generate perplexity files for the test set
            test_file_path = f"output/2022101094_{lm_name}_{n}_test_perplexity_{dataset}.txt"
            generate_perplexity_file(lm, test_corpus, test_file_path)
            print(f"Generated {test_file_path}")


if __name__ == "__main__":
    main()
