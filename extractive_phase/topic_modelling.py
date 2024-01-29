import pandas as pd
import nltk
nltk.download("stopwords")

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come','what','how','much','year','2017','2018','2019','2020','2021','2022','q1','q2','q3','q4'])

from sklearn.feature_extraction.text import CountVectorizer

documents =open('./FLAN_FinBPS/Question_sorted_acc_to_keywords.csv').readlines()
doc =[]
for d in documents:
  d= d.strip()
  if(len(d)>0):
    doc.append(d)
documents = doc

# Create a CountVectorizer to convert text to a document-term matrix
vectorizer = CountVectorizer(strip_accents = 'unicode',
                                stop_words = 'english'
                               )
document_term_matrix = vectorizer.fit_transform(documents)

# Convert the document-term matrix to a dense array for display
dense_matrix = document_term_matrix.toarray()

# Display the document-term matrix as a DataFrame

# Create a DataFrame with term names as columns and documents as rows
df = pd.DataFrame(data=dense_matrix, columns=vectorizer.get_feature_names_out(), index=range(1, len(documents) + 1))

print("Document-Term Matrix:")
print(df)

# Parameters tuning using Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
grid_params = {'n_components' : list(range(5,10))}
# LDA model
lda = LatentDirichletAllocation()
lda_model = GridSearchCV(lda,param_grid=grid_params)
lda_model.fit(document_term_matrix)
# Estimators for LDA model
lda_model1 = lda_model.best_estimator_
print("Best LDA model's params" , lda_model.best_params_)
print("Best log likelihood Score for the LDA model",lda_model.best_score_)
print("LDA model Perplexity on train data", lda_model1.perplexity(document_term_matrix))



n_topics = 30  #specify the number of topics
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(document_term_matrix)

# Get the topic-word probabilities
topic_word_probabilities = lda.components_
topiclist=[]
# Print the top words for each topic
n_top_words = 3 # Number of top words to display for each topic
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(topic_word_probabilities):
    top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    dict = {"Topic" +str(topic_idx + 1):', '.join(top_words) }
    topiclist.append(dict)