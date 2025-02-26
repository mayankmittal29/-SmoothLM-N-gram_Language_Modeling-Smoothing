# import matplotlib.pyplot as plt
# import sys
# import math
# import numpy as np


# def extract_average_perplexity(file_path):
#     with open(file_path, "r") as f:
#         first_line = f.readline().strip()
#         avg_perplexity = float(first_line.split(": ")[1])
#     return avg_perplexity

# def collect_perplexity_scores(dataset):
#     lm_types = ["laplace", "good_turing", "interpolation"]
#     n_values = [1, 3, 5]
#     perplexity_scores = {}

#     for lm_name in lm_types:
#         for n in n_values:
#             # Generate file paths for train and test sets
#             train_file_path = (
#                 f"output/2022101094_{lm_name}_{n}_train_perplexity_{dataset}.txt"
#             )
#             test_file_path = (
#                 f"output/2022101094_{lm_name}_{n}_test_perplexity_{dataset}.txt"
#             )

#             # Extract average perplexity scores
#             train_perplexity = extract_average_perplexity(train_file_path)
#             test_perplexity = extract_average_perplexity(test_file_path)

#             # Store scores in the dictionary
#             key = f"{lm_name}_n{n}"
#             perplexity_scores[key] = {
#                 "train": train_perplexity,
#                 "test": test_perplexity,
#             }

#     return perplexity_scores

# def plot_perplexity_comparison(perplexity_scores, dataset):
#     lm_types = ["laplace", "good_turing", "interpolation"]
#     n_values = [1, 3, 5]

#     train_scores = []
#     test_scores = []
#     labels = []

#     for lm_name in lm_types:
#         for n in n_values:
#             key = f"{lm_name}_n{n}"
#             # Apply logarithm to perplexity values
#             train_scores.append(math.log(perplexity_scores[key]["train"]))
#             test_scores.append(math.log(perplexity_scores[key]["test"]))
#             labels.append(f"{lm_name}\nn={n}")

#     # Create bar positions
#     x = np.arange(len(labels))
#     width = 0.35  # Bar width

#     # Plot bars
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.bar(x - width / 2, train_scores, width, label="Train (log)", color="skyblue")
#     ax.bar(x + width / 2, test_scores, width, label="Test (log)", color="orange")

#     # Customize plot
#     ax.set_xlabel("Language Model (n-value)", fontsize=12)
#     ax.set_ylabel("Log Perplexity", fontsize=12)
#     ax.set_title(f"Log Perplexity Comparison for {dataset.capitalize()} Dataset", fontsize=14)
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels, rotation=45, ha="right")
#     ax.legend(fontsize=10)
#     ax.grid(axis="y", linestyle="--", alpha=0.7)

#     # Adjust layout
#     plt.tight_layout()
#     plt.show()




# def main():
#     if len(sys.argv) < 2:
#         print("Usage: python perplex.py <corpus_path>")
#         sys.exit(1)

#     corpus_path = sys.argv[1]

#     if "pride" in corpus_path or "Pride" in corpus_path:
#         dataset = "pride"
#     elif "Ulysses" in corpus_path or "ulysses" in corpus_path:
#         dataset = "ulysses"

#     perplexity_scores = collect_perplexity_scores(dataset)
#     plot_perplexity_comparison(perplexity_scores, dataset)

# if __name__ == "__main__":
#     main()
import matplotlib.pyplot as plt
import sys
import math
import numpy as np
import os
import seaborn as sns
def extract_average_perplexity(file_path):
    with open(file_path, "r") as f:
        first_line = f.readline().strip()
        avg_perplexity = float(first_line.split(": ")[1])
    return avg_perplexity

# def extract_sentence_perplexities(file_path):
#     with open(file_path, "r") as f:
#         lines = f.readlines()[1:]  # Skip the first line (average perplexity)
#         sentence_perplexities = [float(line.strip()) for line in lines]
#     return sentence_perplexities
def extract_sentence_perplexities(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()[1:]  # Skip the first line
        # Extract only the numerical value after the tab character
        sentence_perplexities = [float(line.strip().split("\t")[1]) for line in lines if "\t" in line]
    return sentence_perplexities

def collect_perplexity_scores(dataset):
    lm_types = ["laplace", "good_turing", "interpolation"]
    n_values = [1, 3, 5]
    perplexity_scores = {}

    for lm_name in lm_types:
        for n in n_values:
            # Generate file paths for train and test sets
            train_file_path = (
                f"output/2022101094_{lm_name}_{n}_train_perplexity_{dataset}.txt"
            )
            test_file_path = (
                f"output/2022101094_{lm_name}_{n}_test_perplexity_{dataset}.txt"
            )

            # Extract average perplexity scores
            train_perplexity = extract_average_perplexity(train_file_path)
            test_perplexity = extract_average_perplexity(test_file_path)

            # Extract sentence perplexities
            train_sentence_perplexities = extract_sentence_perplexities(train_file_path)
            test_sentence_perplexities = extract_sentence_perplexities(test_file_path)

            # Store scores in the dictionary
            key = f"{lm_name}_n{n}"
            perplexity_scores[key] = {
                "train": train_perplexity,
                "test": test_perplexity,
                "train_sentence_perplexities": train_sentence_perplexities,
                "test_sentence_perplexities": test_sentence_perplexities,
            }

    return perplexity_scores

# def collect_perplexity_scores(dataset):
#     lm_types = ["laplace", "good_turing", "interpolation"]
#     n_values = [1, 3, 5]
#     perplexity_scores = {}

#     for lm_name in lm_types:
#         for n in n_values:
#             # Generate file paths for train and test sets
#             train_file_path = (
#                 f"output/2022101094_{lm_name}_{n}_train_perplexity_{dataset}.txt"
#             )
#             test_file_path = (
#                 f"output/2022101094_{lm_name}_{n}_test_perplexity_{dataset}.txt"
#             )

#             # Extract average perplexity scores
#             train_perplexity = extract_average_perplexity(train_file_path)
#             test_perplexity = extract_average_perplexity(test_file_path)

#             # Extract sentence-level perplexities
#             train_sentence_perplexities = extract_sentence_perplexities(train_file_path)
#             test_sentence_perplexities = extract_sentence_perplexities(test_file_path)

#             # Store scores in the dictionary
#             key = f"{lm_name}_n{n}"
#             perplexity_scores[key] = {
#                 "train": train_perplexity,
#                 "test": test_perplexity,
#                 "train_sentences": train_sentence_perplexities,
#                 "test_sentences": test_sentence_perplexities,
#             }

#     return perplexity_scores

def save_plot(fig, plot_name):
    output_dir = "images"
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, plot_name), dpi=300)

def plot_perplexity_comparison(perplexity_scores, dataset):
    lm_types = ["laplace", "good_turing", "interpolation"]
    n_values = [1, 3, 5]

    train_scores = []
    test_scores = []
    labels = []

    for lm_name in lm_types:
        for n in n_values:
            key = f"{lm_name}_n{n}"
            # Apply logarithm to perplexity values
            train_scores.append(math.log(perplexity_scores[key]["train"]))
            test_scores.append(math.log(perplexity_scores[key]["test"]))
            labels.append(f"{lm_name}\nn={n}")

    # Create bar positions
    x = np.arange(len(labels))
    width = 0.35  # Bar width

    # Plot bars
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, train_scores, width, label="Train (log)", color="skyblue")
    ax.bar(x + width / 2, test_scores, width, label="Test (log)", color="orange")

    # Customize plot
    ax.set_xlabel("Language Model (n-value)", fontsize=12)
    ax.set_ylabel("Log Perplexity", fontsize=12)
    ax.set_title(f"Log Perplexity Comparison for {dataset.capitalize()} Dataset", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Adjust layout
    plt.tight_layout()
    save_plot(fig, f"log_perplexity_comparison_{dataset}.png")
    plt.show()

# def plot_sentence_perplexity_distribution(perplexity_scores, dataset):
#     fig, axes = plt.subplots(3, 2, figsize=(12, 12))
#     axes = axes.flatten()

#     for idx, (key, values) in enumerate(perplexity_scores.items()):
#         ax = axes[idx]
#         ax.hist(values["train_sentences"], bins=30, alpha=0.7, label="Train", color="blue")
#         ax.hist(values["test_sentences"], bins=30, alpha=0.7, label="Test", color="red")
#         ax.set_title(f"{key}")
#         ax.set_xlabel("Perplexity")
#         ax.set_ylabel("Frequency")
#         ax.legend()

#     plt.tight_layout()
#     save_plot(fig, f"sentence_perplexity_distribution_{dataset}.png")
#     plt.show()
def plot_sentence_perplexity_distribution(perplexity_scores, dataset):
    lm_types = ["laplace", "good_turing", "interpolation"]
    n_values = [1, 3, 5]

    # Calculate total number of plots needed
    total_plots = len(lm_types) * len(n_values)

    # Create subplots dynamically based on total_plots
    cols = 3  # Fixed number of columns
    rows = math.ceil(total_plots / cols)  # Calculate rows based on total plots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

    # Flatten axes for easy indexing
    axes = axes.flatten()

    for idx, (lm_name, n) in enumerate([(lm, n) for lm in lm_types for n in n_values]):
        key = f"{lm_name}_n{n}"
        sentence_perplexities = perplexity_scores[key]["train_sentence_perplexities"]

        # Plot histogram
        max_perplexity = max(sentence_perplexities)
        sns.histplot(sentence_perplexities, bins=20, kde=True, color="skyblue", ax=axes[idx], stat="density", alpha=0.7)
        axes[idx].hist(sentence_perplexities, bins=20, color="skyblue", alpha=0.7)
        axes[idx].set_title(f"{lm_name.capitalize()} (n={n})", fontsize=12)
        axes[idx].set_xlabel("Sentence Perplexity", fontsize=10)
        axes[idx].set_ylabel("Frequency", fontsize=10)
        axes[idx].grid(alpha=0.3)
        axes[idx].set_xlim(0, max_perplexity)
    # Hide any unused subplots
    for ax in axes[total_plots:]:
        ax.axis("off")

    # Add a title for the entire figure
    fig.suptitle(f"Sentence Perplexity Distribution for {dataset.capitalize()} Dataset", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the figure to the images folder
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/sentence_perplexity_distribution_{dataset}.png")

    # Show the plot
    plt.show()
def plot_boxplots(perplexity_scores, dataset):
    lm_types = ["laplace", "good_turing", "interpolation"]
    n_values = [1, 3, 5]

    train_data = []
    test_data = []
    labels = []

    for lm_name in lm_types:
        for n in n_values:
            key = f"{lm_name}_n{n}"
            if "train_sentence_perplexities" in perplexity_scores[key]:
                train_data.append(perplexity_scores[key]["train_sentence_perplexities"])
            if "test_sentence_perplexities" in perplexity_scores[key]:
                test_data.append(perplexity_scores[key]["test_sentence_perplexities"])
            labels.append(f"{lm_name}\nn={n}")

    # Create subplots for train and test boxplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Train boxplot
    axes[0].boxplot(train_data, labels=labels, patch_artist=True)
    axes[0].set_title(f"Train Sentence Perplexities for {dataset.capitalize()} Dataset")
    axes[0].set_xlabel("Language Model (n-value)")
    axes[0].set_ylabel("Perplexity")
    axes[0].set_ylim(50, 10000)  # Set y-axis range
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # Test boxplot
    axes[1].boxplot(test_data, labels=labels, patch_artist=True)
    axes[1].set_title(f"Test Sentence Perplexities for {dataset.capitalize()} Dataset")
    axes[1].set_xlabel("Language Model (n-value)")
    axes[1].set_ylim(50, 1_000_000)  # Set y-axis range
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    # Adjust layout and save the plot
    plt.tight_layout()
    output_path = f"images/boxplot_sentence_perplexities_{dataset}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Boxplots saved to {output_path}")

# def plot_boxplots(perplexity_scores, dataset):
#     lm_types = ["laplace", "good_turing", "interpolation"]
#     n_values = [1, 3, 5]

#     train_data = []
#     test_data = []
#     labels = []

#     for lm_name in lm_types:
#         for n in n_values:
#             key = f"{lm_name}_n{n}"
#             if "train_sentence_perplexities" in perplexity_scores[key]:
#                 train_data.append(perplexity_scores[key]["train_sentence_perplexities"])
#             if "test_sentence_perplexities" in perplexity_scores[key]:
#                 test_data.append(perplexity_scores[key]["test_sentence_perplexities"])
#             labels.append(f"{lm_name}\nn={n}")

#     # Create subplots for train and test boxplots
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

#     # Train boxplot
#     axes[0].boxplot(train_data, labels=labels, patch_artist=True)
#     axes[0].set_title(f"Train Sentence Perplexities for {dataset.capitalize()} Dataset")
#     axes[0].set_xlabel("Language Model (n-value)")
#     axes[0].set_ylabel("Perplexity")
#     axes[0].grid(axis="y", linestyle="--", alpha=0.7)

#     # Test boxplot
#     axes[1].boxplot(test_data, labels=labels, patch_artist=True)
#     axes[1].set_title(f"Test Sentence Perplexities for {dataset.capitalize()} Dataset")
#     axes[1].set_xlabel("Language Model (n-value)")
#     axes[1].grid(axis="y", linestyle="--", alpha=0.7)

#     # Adjust layout and save the plot
#     plt.tight_layout()
#     output_path = f"images/boxplot_sentence_perplexities_{dataset}.png"
#     plt.savefig(output_path)
#     plt.close()
#     print(f"Boxplots saved to {output_path}")

# def plot_boxplots(perplexity_scores, dataset):
#     train_data = []
#     test_data = []
#     labels = []

#     for key, values in perplexity_scores.items():
#         train_data.append(values["train_sentences"])
#         test_data.append(values["test_sentences"])
#         labels.append(key)

#     fig, ax = plt.subplots(figsize=(12, 6))
#     ax.boxplot(train_data, positions=np.arange(len(labels)) * 2.0 - 0.4, widths=0.6, patch_artist=True, boxprops=dict(facecolor="skyblue"))
#     ax.boxplot(test_data, positions=np.arange(len(labels)) * 2.0 + 0.4, widths=0.6, patch_artist=True, boxprops=dict(facecolor="orange"))

#     ax.set_xticks(np.arange(len(labels)) * 2.0)
#     ax.set_xticklabels(labels, rotation=45, ha="right")
#     ax.set_title(f"Sentence Perplexity Boxplots for {dataset.capitalize()} Dataset")
#     ax.set_xlabel("Language Model")
#     ax.set_ylabel("Perplexity")
#     plt.tight_layout()
#     save_plot(fig, f"boxplots_perplexity_{dataset}.png")
#     plt.show()

def plot_heatmap(perplexity_scores, dataset):
    import seaborn as sns

    lm_types = ["laplace", "good_turing", "interpolation"]
    n_values = [1, 3, 5]

    heatmap_data = []
    for lm_name in lm_types:
        row = []
        for n in n_values:
            key = f"{lm_name}_n{n}"
            avg_score = (perplexity_scores[key]["train"] + perplexity_scores[key]["test"]) / 2
            row.append(avg_score)
        heatmap_data.append(row)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=n_values, yticklabels=lm_types, ax=ax)
    ax.set_title(f"Average Perplexity Heatmap for {dataset.capitalize()} Dataset")
    ax.set_xlabel("n-value")
    ax.set_ylabel("Language Model")
    plt.tight_layout()
    save_plot(fig, f"heatmap_perplexity_{dataset}.png")
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python perplex.py <corpus_path>")
        sys.exit(1)

    corpus_path = sys.argv[1]

    if "pride" in corpus_path or "Pride" in corpus_path:
        dataset = "pride"
    elif "Ulysses" in corpus_path or "ulysses" in corpus_path:
        dataset = "ulysses"

    perplexity_scores = collect_perplexity_scores(dataset)
    plot_perplexity_comparison(perplexity_scores, dataset)
    plot_sentence_perplexity_distribution(perplexity_scores, dataset)
    plot_boxplots(perplexity_scores, dataset)
    plot_heatmap(perplexity_scores, dataset)

if __name__ == "__main__":
    main()
