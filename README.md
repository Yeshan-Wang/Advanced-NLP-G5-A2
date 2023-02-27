# Advanced-NLP-G5-A2
The code was carried out by Yeshan Wang and Payam Fakhraie during the course ‘Advanced NLP' taught by Luis Morgado da Costa and Jose Angel Daza at VU Amsterdam.

## REQUIREMENTS
- Pandas 1.5
- Sklearn 1.2

## Code
All code and notebooks should be run in the following order:

### preprocessing.ipynb
The notebook split each sentence into propositions based on predicates from the original data (so each instance has a single labeled argument structure) and save as corresponding conll files in the data directory:
- data/preprocessed data/train.conll
- data/preprocessed data/test.conll

### main.py
1.The script extracts features and labels for argument identification from the preprocessed data by calling several functions from feature_extraction.py and save as corresponding conll files in the data directory:
- data/features and gold label for argument identification/train.conll
- data/features and gold label for argument identification/test.conll



