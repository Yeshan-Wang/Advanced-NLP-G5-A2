from feature_extraction import read_conll_file
from feature_extraction import extract_features_and_labels
from feature_extraction import extract_features_and_gold_labels
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics
import pandas as pd

## Argument identification

# Extract features and gold label for argument identification from preprocessed training data
train = extract_features_and_labels('data/preprocessed data/train.conll')
train_features = train.drop(["argument_label"], axis=1)
train_labels = train["argument_label"].copy()
vectorizer = DictVectorizer()
train_features_vectorized = vectorizer.fit_transform(train_features.to_dict('records'))

# Extract features and gold label for argument identification from preprocessed test data
test = extract_features_and_labels('data/preprocessed data/test.conll')
test_features = test.drop(["argument_label"], axis=1)
test_labels = test["argument_label"].copy()
test_features_vectorized = vectorizer.transform(test_features.to_dict('records'))

# argument identification with Logistic Regression model
classifier = LogisticRegression()
classifier.fit(train_features_vectorized, train_labels)
predictions = classifier.predict(test_features_vectorized)

# get evaluation results of argument identification
report = classification_report(test_labels, predictions, digits=3)
print('Argument identification with Logistic Regression model:')
print('')
print(report)

## Argument classification

# extract instances that have a “valid argument” label in training set
df = read_conll_file('data/preprocessed data/train.conll')
df['argument_label'] = train['argument_label']
result = df.loc[df['argument_label'] == 'valid argument']
trainset = result.drop(["argument_label"], axis=1)  # replace the binary labels with the original gold labels for argument classification
trainset.to_csv('data/argument classification/train.conll', sep='\t', index=False)  # save as corresponding conll files for argument classification

# extract features and labels from the corresponding training set for argument classification
trainset = extract_features_and_gold_labels('data/argument classification/train.conll')
train_features = trainset.drop(["label"], axis=1)
train_labels = trainset["label"].copy()
vectorizer = DictVectorizer()
train_features_vectorized = vectorizer.fit_transform(train_features.to_dict('records'))

# extract instances that have been predicted as a “valid argument” label in test set
df = read_conll_file('data/preprocessed data/test.conll')
df['argument_label'] = predictions.tolist()
result = df.loc[df['argument_label'] == 'valid argument']
testset = result.drop(["argument_label"], axis=1)  # replace the binary labels with the original gold labels for argument classification
testset.to_csv('data/argument classification/test.conll', sep='\t', index=False)    # save as corresponding conll files for argument classification

# extract features and labels from the corresponding test set for argument classification
testset = extract_features_and_gold_labels('data/argument classification/test.conll')
test_features = testset.drop(["label"], axis=1)
test_labels = testset["label"].copy()
test_features_vectorized = vectorizer.transform(test_features.to_dict('records'))

# argument classification with Logistic Regression model
classifier = LogisticRegression()
classifier.fit(train_features_vectorized, train_labels)
predictions = classifier.predict(test_features_vectorized)

# get evaluation results of argument classification
report = classification_report(test_labels, predictions, digits=3, zero_division=0)
print('Argument classification with Logistic Regression model:')
print('')
print(report)

# Save the final output of the system (i.e the predictions) on the test set 
df = pd.read_csv('data/argument classification/test.conll', sep='\t')
df['predictions'] = predictions.tolist()
df.to_csv('data/system output/predictions on testset.conll', sep='\t', index=False)  