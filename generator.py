import sys
from tokenizer import tokenize_text
import heapq
import pickle
import os
from language_model import (
    LaplaceLanguageModel,
    GoodTuringLanguageModel,
    InterpolationLanguageModel,
    NoSmoothingLanguageModel,
)


class TextGenerator:
    def __init__(self, model):
        self.model = model

    def predict_next_word(self, sentence, top_k=3):
        sentence = ["<s>"] * (self.model.n - 1) + sentence
        context = tuple(sentence[-(self.model.n - 1) :])
        probabilities = []
        for word in self.model.vocabulary:
            ngram = context + (word,)
            # print(ngram)
            prob = self.model.get_probability(ngram)
            probabilities.append((word, prob))

        top_predictions = heapq.nlargest(top_k, probabilities, key=lambda x: x[1])

        return top_predictions


def save_model(model, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(model, file)


def load_model(file_path):
    with open(file_path, "rb") as file:
        model = pickle.load(file)
    return model

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python generator.py <lm_type> <corpus_path> <k> <n>")
        sys.exit(1)

    lm_type = sys.argv[1]
    corpus_path = sys.argv[2]
    k = int(sys.argv[3])
    n = int(sys.argv[4])

    # Check if the model is Good-Turing and if a saved model exists
    if lm_type == "g":
        model_file = f"good_turing_model_{os.path.basename(corpus_path)}_n{n}.pkl"
        if os.path.exists(model_file):
            # print("Loading pre-trained Good-Turing model from file...")
            model = load_model(model_file)
        else:
            # print("Training Good-Turing model...")
            model = GoodTuringLanguageModel(n=n)
            # Load and tokenize the corpus
            corpus = []
            with open(corpus_path, "r") as corpus_file:
                corpus_content = corpus_file.read()
            corpus_content = tokenize_text(corpus_content)
            for sentence in corpus_content:
                corpus.append(sentence)
            # Train the model
            model.train(corpus)
            # Save the trained model
            save_model(model, model_file)
            print("Good-Turing model trained and saved successfully!")
    else:
        if lm_type == "l":
            model = LaplaceLanguageModel(n=n)
        elif lm_type == "i":
            model = InterpolationLanguageModel(n=n)
        elif lm_type == "n":
            model = NoSmoothingLanguageModel(n=n)
        else:
            print("Invalid lm_type. Choose l, g, i, or n.")
            sys.exit(1)

        # Load and tokenize the corpus
        corpus = []
        with open(corpus_path, "r") as corpus_file:
            corpus_content = corpus_file.read()
        corpus_content = tokenize_text(corpus_content)
        for sentence in corpus_content:
            corpus.append(sentence)
        # Train the model
        model.train(corpus)
        print("Language model trained successfully!")

    # Create a text generator instance
    generator = TextGenerator(model)

    # Get input sentence for prediction
    input_sentence = input("input sentence: ")
    tokenized_sentence = tokenize_text(input_sentence)[0]
    # Predict next word
    predictions = generator.predict_next_word(tokenized_sentence, top_k=k)

    print(f"output:")
    for word, prob in predictions:
        print(f"{word} \t {prob}")
