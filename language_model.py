from collections import defaultdict, Counter
import math
import sys
from tokenizer import tokenize_text
from sklearn.linear_model import LinearRegression
import numpy as np


class LaplaceLanguageModel:
    def __init__(self, n=1):
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocabulary = set()
        self.total_count = 0

    def train(self, corpus):
        for sentence in corpus:
            sentence = ["<s>"] * (self.n - 1) + sentence + ["</s>"]
            # print(sentence)
            for i in range(len(sentence) - self.n + 1):
                ngram = tuple(sentence[i : i + self.n])
                context = tuple(sentence[i : i + self.n - 1])
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1
                self.vocabulary.update(ngram)

        self.vocab_size = len(self.vocabulary)
        self.total_count = sum(self.ngram_counts.values())

    def get_probability(self, ngram):
        if self.n == 1:
            context = ()
            ngram = ngram[-1:]
            # print(ngram)
            word = tuple([ngram])
            ngram_count = self.ngram_counts[word]
            return (ngram_count + 1) / (self.total_count + self.vocab_size)
        else:
            context = tuple(ngram[:-1])
            ngram_count = self.ngram_counts[ngram]
            context_count = self.context_counts[context]
            return (ngram_count + 1) / (context_count + self.vocab_size)

    def get_sentence_probability(self, sentence):
        sentence = ["<s>"] * (self.n - 1) + sentence + ["</s>"]
        log_probability = 0.0

        for i in range(len(sentence) - self.n + 1):
            ngram = tuple(sentence[i : i + self.n])
            probability = self.get_probability(ngram)
            log_probability += math.log(probability)

        return log_probability


class GoodTuringLanguageModel:
    def __init__(self, n=1):
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocabulary = set()
        self.adjusted_counts = {}
        self.total_count = 0

    def train(self, corpus):
        for sentence in corpus:
            sentence = ["<s>"] * (self.n - 1) + sentence + ["</s>"]
            for i in range(len(sentence) - self.n + 1):
                ngram = tuple(sentence[i : i + self.n])
                context = tuple(sentence[i : i + self.n - 1])
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1
                self.vocabulary.update(ngram)

        self.vocab_size = len(self.vocabulary)
        self._calculate_adjusted_counts()
        self.total_count = sum(self.ngram_counts.values())

    def _calculate_adjusted_counts(self):
        count_of_counts = Counter(self.ngram_counts.values())
        max_count = max(self.ngram_counts.values())

        x = []
        y = []
        for c in range(1, max_count + 1):
            if count_of_counts[c] > 0:
                x.append(math.log(c))
                y.append(math.log(count_of_counts[c]))

        x = np.array(x).reshape(-1, 1)
        y = np.array(y)

        model = LinearRegression()
        model.fit(x, y)

        smoothed_counts = {}
        for c in range(max_count + 1):
            if c > 3:
                log_c = math.log(c)
                log_c_next = math.log(c + 1)
                n_c = math.exp(model.predict([[log_c]])[0])
                n_c_next = math.exp(model.predict([[log_c_next]])[0])
                smoothed_counts[c] = (c + 1) * n_c_next / n_c
            else:
                smoothed_counts[c] = c

        self.adjusted_counts = smoothed_counts

    def get_probability(self, ngram):
        count = self.ngram_counts[ngram]
        context = tuple(ngram[:-1])
        context_count = self.context_counts[context]

        if self.n == 1:
            context_count = self.total_count
            word = ngram[-1]
            word = tuple([word])
            count = self.ngram_counts[word]

        if count in self.adjusted_counts:
            adjusted_count = self.adjusted_counts[count]
        else:
            adjusted_count = count

        if adjusted_count > 0:
            return adjusted_count / context_count

        else:
            n1 = self.adjusted_counts.get(1, 1)
            return n1 / self.total_count

    def get_sentence_probability(self, sentence):
        sentence = ["<s>"] * (self.n - 1) + sentence + ["</s>"]
        log_probability = 0.0

        for i in range(len(sentence) - self.n + 1):
            ngram = tuple(sentence[i : i + self.n])
            probability = self.get_probability(ngram)
            if probability > 0:
                log_probability += math.log(probability)
        return log_probability



class InterpolationLanguageModel:

    def __init__(self, n=3):
        self.n = n
        self.ngram_counts = [defaultdict(int) for _ in range(n)]
        self.context_counts = [defaultdict(int) for _ in range(n)]
        self.vocabulary = set()
        self.lambdas = [0.0] * n

    def train(self, corpus):
        for sentence in corpus:
            for k in range(self.n):
                padded_sentence = ["<s>"] * k + sentence + ["</s>"]
                # if k != 0:
                #     padded_sentence += ["</s>"]
                for i in range(len(padded_sentence) - k):
                    ngram = tuple(padded_sentence[i : i + k + 1])
                    context = tuple(padded_sentence[i : i + k])
                    # if k == 0:
                    #     print(self.context_counts[k])
                    self.ngram_counts[k][ngram] = self.ngram_counts[k].get(ngram, 0) + 1
                    self.context_counts[k][context] = self.context_counts[k].get(context, 0) + 1
                    self.vocabulary.update(ngram)

        self.vocab_size = len(self.vocabulary)
        self._calculate_lambdas()

    def _calculate_lambdas(self):
        lambda_counts = [0.0] * self.n

        for ngram, count in self.ngram_counts[self.n - 1].items():
            max_count = -1
            corresponding_idx = -1

            for k in range(1,self.n):
                context = ngram[:k]
                ngram_k = ngram[: k + 1]
                # print((self.context_counts[0]))
                count_gram = self.ngram_counts[k].get(ngram_k, 0)
                count_context = self.context_counts[k].get(context, 0)
                # print(f"k: {k} \t {ngram_k}:{count_gram} \t {context}:{count_context}")
                if count_context > 0:
                    if count_gram > max_count:
                        max_count = count_gram
                        corresponding_idx = k

            if corresponding_idx != -1:
                lambda_counts[corresponding_idx] += count

        total = 0
        for count in lambda_counts:
            total += count
        if total > 0:
            # self.lambdas = [count / total for count in lambda_counts]
            for i in range(0,self.n):
                self.lambdas[i]=lambda_counts[i]/total

        # print("Lambdas:", self.lambdas)
        return self.lambdas


    def get_probability(self, ngram):
        probability = 0.0
        for k in range(self.n):
            context = ngram[:k]
            ngram_k = ngram[: k + 1]
            if self.context_counts[k][context] > 0:
                probability += self.lambdas[k] * (
                    self.ngram_counts[k][ngram_k] / self.context_counts[k][context]
                )
            else:
                probability += 0.0  # Unseen context

        return probability

    def get_sentence_probability(self, sentence):
        # Pad the sentence for the highest order n-gram
        padded_sentence = ["<s>"] * (self.n - 1) + sentence + ["</s>"]
        log_probability = 0.0

        for i in range(len(padded_sentence) - self.n + 1):
            ngram = tuple(padded_sentence[i : i + self.n])
            probability = self.get_probability(ngram)
            if probability > 0:
                log_probability += math.log(probability)
            else:
                log_probability += math.log(
                    1 / self.vocab_size
                )  # Handle unseen n-grams

        return log_probability


class NoSmoothingLanguageModel:
    def __init__(self, n=1):
        self.n = n
        self.ngram_counts = defaultdict(int)  # Stores counts of n-grams
        self.context_counts = defaultdict(int)  # Stores counts of contexts
        self.vocabulary = set()  # Stores the vocabulary
        self.total_count = 0  # Total count of n-grams

    def train(self, corpus):
        for sentence in corpus:
            sentence = ["<s>"] * (self.n - 1) + sentence + ["</s>"]
            for i in range(len(sentence) - self.n + 1):
                ngram = tuple(sentence[i : i + self.n])
                context = tuple(sentence[i : i + self.n - 1])
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1
                self.vocabulary.update(ngram)

        self.vocab_size = len(self.vocabulary)
        self.total_count = sum(self.ngram_counts.values())

    def get_probability(self, ngram):
        context = tuple(ngram[:-1])
        ngram_count = self.ngram_counts[ngram]
        context_count = self.context_counts[context]

        if self.n == 1:
            context_count = self.total_count
            word = ngram[-1]
            word = tuple([word])
            ngram_count = self.ngram_counts[word]
        # print(ngram,context_count,ngram_count)
        if context_count == 0:
            return 0.0  # Unseen context
        return ngram_count / context_count

    def get_sentence_probability(self, sentence):
        sentence = ["<s>"] * (self.n - 1) + sentence + ["</s>"]
        log_probability = 0.0
        epsilon = 1e-10  # Small probability for unseen n-grams

        for i in range(len(sentence) - self.n + 1):
            ngram = tuple(sentence[i : i + self.n])
            probability = self.get_probability(ngram)
            if probability > 0:
                log_probability += math.log(probability)
            else:
                log_probability += math.log(epsilon)  # Handle unseen n-grams

        return log_probability


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python lm1.py <lm_type> <corpus_path> <n>")
        sys.exit(1)

    lm_type = sys.argv[1]
    corpus_path = sys.argv[2]
    n=sys.argv[3]
    try:
        n = int(n)
    except :
        print("Invalid n value. Please enter an integer value.")
        sys.exit(1)


    if lm_type == "l":
        lm = LaplaceLanguageModel(n=n)
    elif lm_type == "g":
        lm = GoodTuringLanguageModel(n=n)
    elif lm_type == "i":
        lm = InterpolationLanguageModel(n=n)
    elif lm_type == "n":
        lm = NoSmoothingLanguageModel(n=n)
    else:
        print(
            "Invalid LM type. Use l for Laplace, g for Good-Turing, or i for Interpolation."
        )
        sys.exit(1)

    corpus = []
    corpus_txt = open(corpus_path, "r")
    corpus_content = corpus_txt.read()
    # print(corpus_content)
    corpus_txt.close()
    corpus_content = tokenize_text(corpus_content)
    for sentence in corpus_content:
        corpus.append(sentence)

    lm.train(corpus)
    print("Language model trained successfully!")

    input_sentence = input("input sentence: ")
    input_sentence = tokenize_text(input_sentence)[0]
    lm_prob = lm.get_sentence_probability(input_sentence)
    prob = math.exp(lm_prob)
    print(f"score: {prob}")
