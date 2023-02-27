from feature_extraction import read_conll_file
from feature_extraction import extract_features_and_labels
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics

# Extract features and gold label for argument identification from preprocessed training data
train = extract_features_and_labels('data/preprocessed data/train.conll')
train.to_csv('data/features and gold label for argument identification/train.conll', sep='\t', index=False)  # Store the output of intermediate step
train_features = train.drop(["argument_label"], axis=1)
train_labels = train["argument_label"].copy()
vectorizer = DictVectorizer()
train_features_vectorized = vectorizer.fit_transform(train_features.to_dict('records'))

# Extract features and gold label for argument identification from preprocessed test data
test = extract_features_and_labels('data/preprocessed data/test.conll')
test.to_csv('data/features and gold label for argument identification/test.conll', sep='\t', index=False)  # Store the output of intermediate step
test_features = test.drop(["argument_label"], axis=1)
test_labels = test["argument_label"].copy()
test_features_vectorized = vectorizer.transform(test_features.to_dict('records'))

# argument identification with Logistic Regression model
classifier = LogisticRegression()
classifier.fit(train_features_vectorized, train_labels)
predictions = classifier.predict(test_features_vectorized)
report = classification_report(test_labels, predictions, digits=3)
print('Argument identification with Logistic Regression model:')
print('')
print(report)

# Extract features and gold label for argument classification of the training set
def prepare_trainset_for_argument_classification(predictions):
    df = read_conll_file('data/preprocessed data/train.conll')
    train['label'] = df['label']
    result = train.loc[train['argument_label'] == 'valid argument']
    results = result.drop(["argument_label"], axis=1)
    return results

trainset = prepare_trainset_for_argument_classification(predictions)
trainset.to_csv('data/features and gold label for argument classification/train.conll', sep='\t', index=False)  # Store the output of intermediate step
train_features = trainset.drop(["label"], axis=1)
train_labels = trainset["label"].copy()
vectorizer = DictVectorizer()
train_features_vectorized = vectorizer.fit_transform(train_features.to_dict('records'))

# Extract features and gold label for argument classification of the test set
def prepare_testset_for_argument_classification(predictions):
    test_features['argument_predictions'] = predictions.tolist()
    df = read_conll_file('data/preprocessed data/test.conll')
    test_features['label'] = df['label']
    result = test_features.loc[test_features['argument_predictions'] == 'valid argument']
    results = result.drop(["argument_predictions"], axis=1)
    return results

testset = prepare_testset_for_argument_classification(predictions)
testset.to_csv('data/features and gold label for argument classification/test.conll', sep='\t', index=False)  # Store the output of intermediate step
test_features = testset.drop(["label"], axis=1)
test_labels = testset["label"].copy()
test_features_vectorized = vectorizer.transform(test_features.to_dict('records'))

# argument classification with Logistic Regression model
classifier = LogisticRegression()
classifier.fit(train_features_vectorized, train_labels)
predictions = classifier.predict(test_features_vectorized)
report = classification_report(test_labels, predictions, digits=3, zero_division=0)
print('Argument classification with Logistic Regression model:')
print('')
print(report)

# Store the system output
testset['argument_predictions'] = predictions.tolist()
testset.to_csv('data/system output/predictions on testset.conll', sep='\t', index=False)