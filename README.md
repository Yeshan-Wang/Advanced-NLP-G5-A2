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
#### 1. extract features and labels for argument identification from the preprocessed data by calling several functions from feature_extraction.py and save as corresponding conll files in the data directory:
- data/features and gold label for argument identification/train.conll
- data/features and gold label for argument identification/test.conll

**There are 6 features selected to carry out the argument identification task:**
- Token
- Lemma of each token
- POS of each token
- Dependency relations
- Lemma of predicate
- Voice of predicate

**Original gold labels are divided into binary labels for argument identification:**
- invalid argument: gold labels that contain keyword of 'ARG'
- valid argument: gold labels that do not contain keyword of 'ARG'

#### 2. Argument identification: train the SVM model with all extract features from the training data, and evaluate it on test data. The result is visible in evaluation_results.ipynb

#### 3. Based on the prediction by the first classifier, extract instances that have been assigned a “valid argument” label and save as corresponding conll files in the data directory:
- data/features and gold label for argument classification/train.conll
- data/features and gold label for argument classification/test.conll

#### 4. Argument classification: train the SVM model with all extract features from the training data, and evaluate it on test data. The result is visible in evaluation_results.ipynb

#### 5. The final output of the system (i.e the predictions) on the test set is saved in the data directory:
- data/system output/predictions on testset.conll

### evaluation_results.ipynb
load and run main.py in jupyter notebook to get evaluation results of argument identification and argument classification


