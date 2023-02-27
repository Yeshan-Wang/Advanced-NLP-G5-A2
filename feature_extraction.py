import pandas as pd

def read_conll_file(inputfile):
    # Read inputfile as pandas dataframe
    df = pd.read_csv(inputfile, sep = '\t', header = None, on_bad_lines='skip', engine='python', 
                     names = ["id", "word", "lemma", "pos-univ", "pos", "morph", "head", "basic_dep", "enhanced_dep" , "space", "predicate", "label"])
    
    # Remove missing values
    df = df.dropna()
    
    return df

def extract_predicate_lemma(row):
    if row["label"] == 'V':
        return row["lemma"]
    else:
        return '_'
    
def extract_voice(row):
    if row["label"] == 'V':
        if 'Voice=Pass' in row["label"]:
            return 'Passive'
        else:
            return 'Active'
    else:
        return '_'

def extract_label(row):
    if 'ARG' in row["label"]:
        return 'valid argument'
    else:
        return 'invalid argument'

def extract_features_and_labels(inputfile):
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