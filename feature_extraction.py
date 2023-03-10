import pandas as pd

def read_conll_file(inputfile):
    '''read the preprocessed data and return a dataframe'''
    # Read inputfile as pandas dataframe
    df = pd.read_csv(inputfile, sep = '\t', header = None, on_bad_lines='skip', engine='python', 
                     names = ["id", "word", "lemma", "pos-univ", "pos", "morph", "head", "basic_dep", "enhanced_dep" , "space", "predicate", "label"])
    # Remove missing values
    df = df.dropna()
    
    return df

def extract_predicate_lemma(row):
    '''extract predicate lemma from each row of the data'''
    if row["label"] == 'V':
        return row["lemma"]
    else:
        return '_'
    
def extract_voice(row):
    '''extract voice from each row of the data'''
    if row["label"] == 'V':
        if 'Voice=Pass' in row["label"]:
            return 'Passive'
        else:
            return 'Active'
    else:
        return '_'

def extract_label(row):
    '''extract label for argument identification from each row of the data'''
    if 'ARG' in row["label"]:
        return 'valid argument'
    else:
        return 'invalid argument'

def extract_features_and_labels(inputfile):
    '''extract features and label for argument identification'''
    # Read inputfile as pandas dataframe
    df = read_conll_file(inputfile)
    
    # create an Empty DataFrame object to store features and labels
    result = pd.DataFrame()
    
    # Feature 1. Token
    result['token'] = df['word']
    
    # Feature 2. Lemma of each token
    result['lemma'] = df['lemma']
    
    # Feature 3. POS of each token
    result['pos'] = df['pos']
    
    # Feature 4. Dependency relations
    result['dep'] = df['basic_dep']
    
    # Feature 5. Lemma of predicate
    result['predicate_lemma'] = df.apply(lambda row: extract_predicate_lemma(row), axis= 1)
    
    # Feature 6. Voice
    result['voice'] = df.apply(lambda row: extract_voice(row), axis= 1)
    
    # extracts gold labels for Argument Identification task
    result['argument_label'] = df.apply(lambda row: extract_label(row), axis= 1)
    
    return result

def extract_features_and_gold_labels_for_training_set(inputfile):
    '''extract features and label for argument classification of the training set'''
    # Read inputfile as pandas dataframe
    df = read_conll_file(inputfile)
    
    # create an Empty DataFrame object to store features and labels
    result = pd.DataFrame()
    
    # Feature 1. Token
    result['token'] = df['word']
    
    # Feature 2. Lemma of each token
    result['lemma'] = df['lemma']
    
    # Feature 3. POS of each token
    result['pos'] = df['pos']
    
    # Feature 4. Dependency relations
    result['dep'] = df['basic_dep']
    
    # Feature 5. Lemma of predicate
    result['predicate_lemma'] = df.apply(lambda row: extract_predicate_lemma(row), axis= 1)
    
    # Feature 6. Voice
    result['voice'] = df.apply(lambda row: extract_voice(row), axis= 1)
    
    # extracts gold labels for Argument Classification task
    result['label'] = df['label']
    
    return result

def extract_features_and_gold_labels_for_test_set(inputfile):
    '''extract features and label for argument classification of the test set'''
    # Read inputfile as pandas dataframe
    df = pd.read_csv(inputfile, sep='\t')
    
    # create an Empty DataFrame object to store features and labels
    result = pd.DataFrame()
    
    # Feature 1. Token
    result['token'] = df['word']
    
    # Feature 2. Lemma of each token
    result['lemma'] = df['lemma']
    
    # Feature 3. POS of each token
    result['pos'] = df['pos']
    
    # Feature 4. Dependency relations
    result['dep'] = df['basic_dep']
    
    # Feature 5. Lemma of predicate
    result['predicate_lemma'] = df.apply(lambda row: extract_predicate_lemma(row), axis= 1)
    
    # Feature 6. Voice
    result['voice'] = df.apply(lambda row: extract_voice(row), axis= 1)
    
    # extracts gold labels for Argument Classification task
    result['label'] = df['label']
    
    return result

