# Introduction TO NLP - Assignment 1
## Important Libraries

1. Install the required libraries using the following command:
```bash
pip install scikit-learn, numpy, matplotlib, seaborn
```
## **Tokenization Schemes**

The tokenizer is designed to handle various patterns and special cases in the text. Below are the tokenization schemes implemented:

1. **Removing `***`**: Any occurrence of `***` is removed from the text.
2. **Removing `—` in front**: Leading dashes (`—`) are removed.
3. **Handling apostrophes**: Words like `preacher's` are retained as single tokens.
4. **Handling dates**: All date formats (e.g., `27th December 2007`, `December 27th, 2007`) are replaced with `<DATE>`.
5. **Retaining hyphenated words**: Words like `knife-blade` are retained as single tokens.
6. **Removing `['.']` in sentences**: Sentences containing only a period are removed.
7. **Abbreviations**: Patterns like `g.p.i` are replaced with `<ABBREVATION>`.
8. **Handling underscores**: Underscores (`_`) are retained only within words (e.g., `chorus excipiat._` becomes `chorus excipiat`).
9. **Removing ellipses (`...`)**: All occurrences of `...` are removed.
10. **Replacing `--` with space**: Double dashes (`--`) are replaced with a single space.
11. **Removing punctuation in parentheses**: Punctuation marks like `?`, `!`, and `.` inside parentheses are removed.
12. **Handling degrees**: Patterns like `45°` are replaced with `<DEGREE>`.
13. **Removing salutations**: All salutations (e.g., `Mr.`, `Mrs.`, `Dr.`) are removed.
14. **Removing double quotes (`""`)**: All double quotes are removed from the text.
15. **Using <>** as placeholders for URLs, hashtags, mentions, percentages, age values, expressions indicating time,
time periods etc

### Use the following command to run the Tokenization code:
```bash
python3 tokenizer.py
```

 ###  Use the following command to train and score the sentence using different language models:
```bash
python3 language_model.py <lm_type> <corpus_path> <n>
```
where:
- `<lm_type>` is the type of language model to be used. It can be one of the following:
    - `l` for Laplace Smoothing
    - `i` for Interpolation Smoothing
    - `g` for Good Turing Smoothing
- `<corpus_path>` is the path to the corpus file
- `<n>` is the value of n-gram to be used
```
Example - python language_model.py l data/corpus.txt 5
```

###  Use the following command to run the next word prediction code:
```bash
python3 generator.py <lm_type> <corpus_path> <k> <n>
``` 
where:
- `<lm_type>` is the type of language model to be used. It can be one of the following:
    - `l` for Laplace Smoothing
    - `g` for Good Turing Smoothing
    - `i` for Interpolation Smoothing
    - `n` for No Smoothing
- `<corpus_path>` is the path to the corpus file
- `<k>` is the number of predictions to be made
- `<n>` is the value of n-gram to be used
```
python3 generator.py l data/corpus.txt 5 5
```
## **To Get Perplexity scores of all sentences**
### Run file "script.py" as 
```
python3 script.py <corpus_path>
```
**By running this file , it will generate 36 .txt files in output folder and avg perplexity scores at top of each file**
### The output of the perplexity files are stored in the format of `{rollnumber}_{lm_type}_{n}_{train/test}_{corpus_name}.txt`

## **To Get plots for both text Corpus of their perplexity values in log scale**
### Run file "graph.py" as 
```
python3 graph.py <corpus_path>
```
**By running this file , it will generate plots with perplexity on y axis in log scale and on x-axis all types of models**
