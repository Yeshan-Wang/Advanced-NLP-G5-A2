# Advanced-NLP-G5-A2
The code was carried out by Yeshan Wang and Payam Fakhraie during the course ‘Advanced NLP' taught by Luis Morgado da Costa and Jose Angel Daza at VU Amsterdam.

## REQUIREMENTS
- Pandas 1.5
- Sklearn 1.2

**All code and notebooks should be run in the following order:**

## preprocessing.ipynb (Estimated code running time: 54s)
The notebook split each sentence into propositions based on predicates from the original data (so each instance has a single labeled argument structure) and save as corresponding conll files in the data directory:
- data/preprocessed data/train.conll
- data/preprocessed data/test.conll

## main.py
The script performs two tasks: Argument identification and Argument classification.

### 1. Argument identification
#### 1.1. extract features and labels for argument identification from the preprocessed test and train data by calling several functions from feature_extraction.py. There are 6 features selected to carry out the argument identification task:
- Token
- Lemma of each token
- POS of each token
- Dependency relations
- Lemma of predicate
- Voice of predicate

**The original gold labels are divided into binary labels for argument identification:**
- invalid argument: gold labels that do not contain keyword of 'ARG'
- valid argument: gold labels that contain keyword of 'ARG'

#### 1.2. train a Logistic Regression classifier with all extract features from the training data, and evaluate it on test data. The evaluation result is visible in evaluation_results.ipynb

### 2. Argument classification
#### 2.1. extract training instances that have a “valid argument” label in training set, replace the binary labels with the original gold labels for argument classification and save as corresponding conll files in the data directory:
- data/argument classification/train.conll

#### 2.2. extract test instances that have been predicted as a “valid argument” label in test set, replace the binary labels with the original gold labels for argument classification and save as corresponding conll files in the data directory:
- data/argument classification/test.conll

#### 2.3. extract features and labels from the corresponding conll files for argument classification. There are 6 features selected to carry out the argument identification task: 
- Token
- Lemma of each token
- POS of each token
- Dependency relations
- Lemma of predicate
- Voice of predicate

#### 2.4. train a second Logistic Regression classifier with all extract features from the training data, and evaluate it on test data. The evaluation result is visible in evaluation_results.ipynb

#### 2.5. The final output of the system (i.e the predictions) on the test set is saved in the data directory:
- data/system output/predictions on testset.conll

## evaluation_results.ipynb (Estimated code running time: 132s)
load and run **main.py** in jupyter notebook to get evaluation results of argument identification and argument classification


