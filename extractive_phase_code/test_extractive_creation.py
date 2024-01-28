import pandas as pd
import numpy as np
import pandas
import pandas as pd
import torch
import pickle
import pandas as pd
from utilities import *

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_recall_fscore_support, classification_report
model = SentenceTransformer('all-MiniLM-L6-v2')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


names_of_files= open('./FLAN_FinBPS/test_files.txt','r+')
fnames = names_of_files.readlines()
for f in fnames:
    f= f.strip()
    output= open('./FLAN_FinBPS/Test/test_files_numerical_lines/'+f,'a+')
    content=[]
    with open('./FLAN_FinBPS/Test/ects/'+f,'r+') as file:
        content = file.readlines()
    lines = ''
    for line in content:
        line = line.strip()
        res = any(chr.isdigit() for chr in line)
        if line!=0 and res== True:
            lines = lines+'\n'+ line
    lines = lines.strip()
    output.write(lines)
  
files = open('./FLAN_FinBPS/test_files.txt','r+')
file_names = files.readlines()
# df = pd.DataFrame(columns=['File','Embedding'])
ect_embeddings = {}
count = 0
for name in file_names:
    print('Processed till ', count)
    content = open('./FLAN_FinBPS/Test/test_files_numerical_lines/'+ name.strip(), 'r+')
    lines = content.readlines()
    all_sent_embedding = model.encode(lines)
    #new_row = {'File': name.strip(), 'Embedding':all_sent_embedding }
    ect_embeddings[name.strip()] =  all_sent_embedding
    #df = df.append(new_row, ignore_index = True)
    count = count+1
    
with open('./FLAN_FinBPS/Test/test_numerical_ect_embeddings.pkl', 'wb') as fp:
    pickle.dump(ect_embeddings, fp)


keyword_file = open('./FLAN_FinBPS/Test/Question_keywords.txt','r+')
keywords= keyword_file.readlines()
file_names= open('./FLAN_FinBPS/test_files.txt','r+')
fnames = file_names.readlines()
df = pd.DataFrame(columns=['Filename','No. of topics covered','Topic names'])
test_topic = {}
for f in fnames:
    f= f.strip()

    content=[]
    with open('./FLAN_FinBPS/Test/test_files_numerical_lines/'+f.strip(),'r+') as file:
        content = file.readlines()
    n=0
    topics=[]
    for key in keywords:
        key = key.strip()
        flag=False
        for line in content:
            if key in line:
                flag=True

        if len(key)>0 and flag:
            n+=1
        topics.append(key)

    new_row= {'Filename': f, 'No. of topics covered': n, 'Topic names': topics }
    test_topic[f.strip()]= topics
    df = df.append(new_row, ignore_index=True)

# print(df)
df.to_csv('./FLAN_FinBPS/Test/test_files_topic_names.csv')
import pickle
with open('./FLAN_FinBPS/Test/test_ect_topiclist.pkl', 'wb') as fp:
    pickle.dump(test_topic, fp)
    
df = pd.read_csv('./FLAN_FinBPS/Question_sorted_acc_to_keywords.csv')
topicwise_encodedquestion={}
topicwise_questions={}
for index, row in df.iterrows():
    topic = row['Keyword']
    questions= row['Question_list']
    questions = questions.strip()
    encoded_questions = model.encode(questions)
    topicwise_encodedquestion[topic.strip()]= encoded_questions
    topicwise_questions[topic.strip()] = questions
print(topicwise_encodedquestion)

import pickle
with open('./FLAN_FinBPS/topic_encodedquestions.pkl', 'wb') as fp:
    pickle.dump(topicwise_encodedquestion, fp)
with open('./FLAN_FinBPS/topicquestions.pkl', 'wb') as fp:
    pickle.dump(topicwise_encodedquestion, fp)
    
def top(Qlist, file_name,q_embedding,f_embedding):
    import numpy as np
    questions=[]
    Qlist_sentences = Qlist.split('\n')
    for q in Qlist_sentences:
        q = q.strip()
        if len(q)>0:
            questions.append(q)
    sentence= open("./FLAN_FinBPS/Test/test_files_numerical_lines/"+file_name.strip(),"r+")
    sentences= sentence.readlines()
    cosine_similarities_pred_all = util.dot_score(q_embedding, f_embedding)
    values, indices = torch.topk(cosine_similarities_pred_all, 5)
    values = values.tolist()
    indices= indices.tolist()
    top_one = top_few(values)
    c1=''
    top_question = questions[top_one].strip()
    try:
        f=0
        for i in range(len(indices[top_one])):
            res = any(chr.isdigit() for chr in sentences[indices[top_one][i]])
            if f >= 3:
                break
            elif res:
                c1 += sentences[indices[top_one][i]].strip() + '\n'
                f += 1
    except Exception as e:
        print(e)
    c1 = c1.strip()

    #print(top_question)
    #print(c1)
    #print(len(c1.split('\n')))
    print(file_name)
    return top_question, np.mean(values[top_one]), c1


import pandas as pd
import pickle
file_names= open('./FLAN_FinBPS/test_files.txt','r+')
fnames = file_names.readlines()
df = pd.read_csv('./FLAN_FinBPS/Question_sorted_acc_to_keywords.csv')
df2 = pd.DataFrame(columns=['File_name','Keyword','Question','Score','Context'])
with open('./FLAN_FinBPS/Test/test_ect_topiclist.pkl', 'rb') as file:
    topic_df = pickle.load(file)
with open('./FLAN_FinBPS/Test/test_numerical_ect_embeddings.pkl', 'rb') as file:
    embedding = pickle.load(file)
with open('./FLAN_FinBPS/topicquestions.pkl', 'rb') as file:
    topic_questions = pickle.load(file)
with open('./FLAN_FinBPS/topic_encodedquestions.pkl', 'rb') as file:
    topic_encodedquestions = pickle.load(file)
count=0
for f in fnames:
    count+=1
    f= f.strip()
    topics = topic_df[f.strip()]
    file_embedding = embedding[f.strip()]

    for topic in topics:
        questions = topic_questions[topic.strip()]
        question_embedding = topic_encodedquestions[topic.strip()]
        top_question,score, context= top(questions,f, question_embedding,file_embedding)
        new_row = {'File_name': f, 'Keyword': topic, 'Question': top_question,'Score':score, 'Context': context}
        df2 = df2.append(new_row, ignore_index = True)
    print('proceeded till '+ str(count))
print(df2)

df2.to_csv('./FLAN_FinBPS/Test/test_numerical_lines_topic_questions_context.csv')


import pandas as pd
# Read the CSV file into a DataFrame
df = pd.read_csv('./FLAN_FinBPS/Test/test_numerical_lines_topic_questions_context.csv')

# Read the file names from the text file
with open('./FLAN_FinBPS/test_files.txt', 'r') as file:
    file_names = [line.strip() for line in file.readlines()]
# file_names=['AA_q3_2021']
# Initialize a list to store dictionaries of data
data_list = []

# Iterate through the file names
for name in file_names:
    # Filter the DataFrame based on the 'File_name' column
    df2 = df[df['File_name'] == name]
    score_list=[]
    for index, row in df2.iterrows():
        if row['Keyword'] != 'revenue' and row['Keyword'] != 'earnings per share':
            score_list.append(row['Score'])
    score_list.sort()
    scores = score_list[-3:]
    counter = 0
    context = ''
    question = ''

    for index, row in df2.iterrows():
        flag = row['Score'] in scores
        if row['Keyword'] == 'revenue' or row['Keyword'] == 'earnings per share' or flag:
            context += '\n' + row['Context']
            question += '\n' + row['Question']
            counter += 1
        if counter == 5:
            print(counter)
            break

    context= unique(context)

    new_data = {
        'File_name': name,
        'Questions': question,
        'Context': context
    }

    data_list.append(new_data)

# Create the final DataFrame from the list of dictionaries
df_final = pd.DataFrame(data_list)
print(df_final)

df_final.to_csv('./FLAN_FinBPS/Combined_test_file_question_context.csv')







