from Data_preprocessing import *
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


# -------------------------------------------- Feature Extraction -----------------------------------------------------

# Write your code here to initialise the TfidfVectorizer
tf_idf_vec = TfidfVectorizer(max_df=0.98,min_df=2,stop_words='english')

# Write your code here to create the Document Term Matrix by transforming the complaints column present in df_clean.
tfidf = tf_idf_vec.fit_transform(df_clean['Complaint_clean'])


# ------------------------------ Topic Modelling using NMF / Manual Topic Modeling -------------------------------------

# Load your nmf_model with the n_components i.e 5
num_topics = 5

# Keep the random_state =40
nmf_model = NMF(n_components=num_topics, random_state=40)

nmf_model.fit(tfidf)
print(len(tf_idf_vec.get_feature_names_out()))


# Print the Top15 words for each of the topics
for index, topic in enumerate(nmf_model.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index} with tf-idf score')
    print([tf_idf_vec.get_feature_names_out()[i] for i in topic.argsort()[-15:]])
    print('\n')


# Create the best topic for each complaint in terms of integer value 0,1,2,3 & 4
topic_values = nmf_model.transform(tfidf)
topic_values.argmax(axis=1)


# Assign the best topic to each of the cmplaints in Topic Column
df_clean['Topic'] = topic_values.argmax(axis=1)


print(tabulate(df_clean.head(), headers='keys', tablefmt='pretty'))

# Print the first 5 Complaint for each of the Topics
df_clean.groupby('Topic').head(5).sort_values(by='Topic')

# Create the dictionary of Topic names and Topics

Topic_names = {
    0: 'Bank Account services',
    1: 'Credit card or prepaid card',
    2: 'Others',
    3: 'Theft/Dispute Reporting',
    4: 'Mortgage/Loan'
}


# Replace Topics with Topic Names
df_clean['Topic_category'] = df_clean['Topic'].map(Topic_names)

print(tabulate(df_clean.head(), headers='keys', tablefmt='pretty'))

# Specify the file path where you want to save the modified DataFrame as a CSV file
output_file = '../../Dataset/Cleaned_Dataset.csv'

# Save the modified DataFrame to a CSV file
df_clean.to_csv(output_file, index=False)

