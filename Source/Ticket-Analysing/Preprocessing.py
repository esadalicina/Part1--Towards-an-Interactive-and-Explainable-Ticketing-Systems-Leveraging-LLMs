import re
import warnings
from tqdm import tqdm
from Analysing_and_Cleaning import *
import spacy

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')
stop_words = nlp.Defaults.stop_words

warnings.filterwarnings('ignore')

# Enable the progress bar for pandas
tqdm.pandas()

# change the display properties of pandas to max
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# ----------------------------------------- Prepare the text for classification ---------------------------------------


def preprocess_text(text):
    text = text.lower()  # Convert to lower case
    text = re.sub(r'^\[[\w\s]\]+$', ' ', text)  # Remove text in square brackets
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = text.replace('\n', '')  # Remove newline characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)   # Remove special characters and numbers
    text = re.sub(r'(.)\1+|\b\w\b', r'\1', text)  # Remove repeated characters and single letters
    
    # Tokenize and filter by POS tags using spaCy
    doc = nlp(text)
    filtered_tokens = [token for token in doc if token.pos_ in {'PRON', 'NOUN', 'VERB', 'ADJ'}]
    
    # Lemmatize the text using spaCy
    lemmatized_text = ' '.join([token.lemma_ for token in filtered_tokens])
    
    return lemmatized_text


# Preprocess text columns
df['complaint_what_happened_lemmatized'] = df['complaint_what_happened'].progress_apply(lambda x: preprocess_text(x)) # type: ignore
# Remove the 'x' 'xx' 'xxx' and 'xxxx' from the text columns
df['complaint_what_happened_lemmatized'] = df['complaint_what_happened_lemmatized'].str.replace(r'x{1,4}', '', regex=True)


# The clean dataframe should now contain the raw and lemmatized complaint with the category and product columns
# print(tabulate(df.head(), headers='keys', tablefmt='pretty')) # type: ignore
print(df['complaint_what_happened_lemmatized'].head(5))

print(df.info())
# Specify the file path where you want to save the modified DataFrame as a CSV file
# output_file = '/home/users/elicina/Master-Thesis/Dataset/Cleaned_Dataset.csv'

# Save the modified DataFrame to a CSV file 
# df.to_csv(output_file, index=False)
