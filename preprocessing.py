import pandas as pd
import spacy

from source import tokenizator_and_preprocessing as tp

"""
1. Preprocess data
"""

# load the dataset from HF
df_full = pd.read_parquet("hf://datasets/ruggsea/stanford-encyclopedia-of-philosophy_chat_multi_turn/data/train-00000-of-00001.parquet")

# work on mock dataset (50 first rows) for writing the code
df = df_full
# Split the conversation into various rows in df
# select utterances of professor
df['professor'] = df['conversation'].apply(lambda x: tp.extract_content(x)[0])

# select utterances of student
df['student'] = df['conversation'].apply(lambda x: tp.extract_content(x)[1])

# clean redundant content from prompt
df['prompt'] = df['prompt'].apply(lambda x: tp.prepare_prompt(x))
df = df.drop('conversation', axis=1)

df = df.explode(["professor", "student"]) # powerful line! creates a pair of student-professor utterance for each row

# explode made unwrapped columns 'professor' and 'student' to strings. For consitency we have to do the sam with 'prompt'
df['prompt'] = df['prompt'].apply(lambda x: ' '.join(map(str, x)))

# df.to_pickle('dfs/preprocessed-df.pkl') # save it for future work

"""
2. Tokenize the data
"""

nlp = spacy.load('en_core_web_sm')
df_clean = tp.tag_df_with_spacy(df=df, nlp=nlp, column_names=['professor', 'student', 'prompt'])

df.to_pickle('dfs/preprocessed-cleaned-df.pkl')
df.to_csv('dfs/preprocessed-cleaned-df.csv')