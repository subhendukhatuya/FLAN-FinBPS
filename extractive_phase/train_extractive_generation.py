import numpy as np
import pandas
import pandas as pd
import torch
from utilities import *

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_recall_fscore_support, classification_report
model = SentenceTransformer('all-MiniLM-L6-v2')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
df2 = pd.DataFrame(
    columns=['File_name','Questionlist', 'Context'])
df3 = pd.DataFrame(
    columns=['File_name','Questionlist', 'Context'])
df = pd.read_csv('./FLAN_FinBPS/File_and_GT_questions.csv')
for index, row in df.iterrows():
    sentence= open("./FLAN_FinBPS/Train/ects/"+row['File_name'].strip(),"r+")
    sentences= sentence.readlines()
    print(row['Question'])
    if(type(row['Question']) == str ):
        questionlist= row['Question'].split('?')
        questionlist = clean(questionlist)
    all_sent_embedding = model.encode(sentences)
    c1=''
    c2=''
    for line in questionlist:
        line = line.strip()
        encoded_line = model.encode(line)
        cosine_similarities_pred_all = util.dot_score(encoded_line, all_sent_embedding)
        values, indices = torch.topk(cosine_similarities_pred_all, 5)
        values = values.tolist()
        indices= indices.tolist()
        f=0
        for i in range(len(indices[0])):
            res = any(chr.isdigit() for chr in sentences[indices[0][i]])
            if res == True:
                if f<1:
                    c1 =c1+ sentences[indices[0][i]]+'\n'
                    c2=c2+ sentences[indices[0][i]]+'\n'
                    f=f+1
                elif f>1 and f<3:
                    c2=c2+ sentences[indices[0][i]]+'\n'
                    f=f+1
    c1= unique(c1)
    c2= unique(c2)
    new_row1= {'File_name': row['File_name'],'Questionlist':row['Question'] , 'Context': c1,}
    df2 = df2.append(new_row1,ignore_index=True)
    new_row2= {'File_name': row['File_name'],'Questionlist':row['Question'] , 'Context': c2,}
    df3 = df3.append(new_row2,ignore_index=True)

            #df=df.append({'Question':question,'Sent':sentences[indices[0][i]],'Score':values[0][i] },ignore_index=True)
        #question_index = range(len(values))
        #top_indices = top_few(values)
        #print(values)
        #print(top_indices)
        #for i in question_index:
        #c=""
        #for j in range(len(indices[0])):
            #res = any(chr.isdigit() for chr in sentences[indices[i][j]])
            #if res == True:
            #c = c+ sentences[indices[i][j]]

    sentence.close()

df2.to_csv('./FLAN_FinBPS/file_question_1sentcontext_gt.csv')
df3.to_csv('./FLAN_FinBPS/question_based_context_train_data.csv')

