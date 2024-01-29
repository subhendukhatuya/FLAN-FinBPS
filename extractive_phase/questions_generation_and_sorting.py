from pipelines import pipeline
import pandas as pd
import nltk
nltk.download('punkt')
nlp = pipeline("question-generation", model="valhalla/t5-small-qg-prepend", qg_format="prepend")

def generate_questions(Sentences):
    Questions=''
    try:
        for s in Sentences:
            s= s.strip()
            #print(s)
            if(len(s)>0):
                res= nlp(s)
            #print(res)
                dictionary = res[0]
            #print(dictionary)
                # print(dictionary)
                question = dictionary['question']
                # print(question)
            #print(question)
                Questions = Questions + ' ' + question
        # print(Questions)
    except Exception as e:
        print(s)
        print("An exception occured:",e)
    return Questions


df = pd.DataFrame(
    columns=['File_name','Question'])
files = open('./FLAN_FinBPS/file_names.txt','r+')
file_names = files.readlines()
for file in file_names:
    file= file.strip()
    gt = './FLAN_FinBPS/Train/gt_summaries/'+ file
    gt_contents=''
    Sentences=[]
    #gt_file = open(gt,'r+')
        # Read the entire contents of the file into a string
    #gt_contents = gt_file.read()
    #gt_file.close()
    gt_file = open(gt,'r+')
    try:
        Sentences = gt_file.readlines()
    except Exception as e:
        print(Sentences)
        print(e)
    # print('new file')
    # print(Sentences)
    # print(gt_contents)
    Questions = generate_questions(Sentences)
    Questions = Questions.strip()
    new_row = {'File_name':file,'Question':Questions}
    df= df.append(new_row, ignore_index= True)
    df.to_csv('./FLAN_FinBPS/File_and_GT_questions.csv')


df = pd.read_csv('./FLAN_FinBPS/File_and_GT_questions.csv')
Questions =[]
for index, row in df.iterrows():
    question = row['Question']
    if(type(question)==str):
        question = row['Question'].split('?')
        for q in question:
            if(len(q)>0 and type(q)==str):
                Questions.append(q.strip())
    print(Questions)
    Questions = list(set(Questions))
    questions_list = open('./FLAN_FinBPS/LIST_OF_QUESTIONS.txt','a')
    c=0
    for question in Questions:        
        if(type(question)==str):
            c=c+1
            questions_list.write(question.strip()+'\n')
    questions_list.close()
    print(c)

import pandas as pd
df=pd.DataFrame(
    columns=['Keyword', 'Question_list','No_of_questions'])
question_file= open(r"./FLAN_FinBPS/LIST_OF_QUESTIONS.txt","r+")
questions= question_file.readlines()
keywords= open(r"./FLAN_FinBPS/Question_keywords.txt","r+")
for keyword in keywords:
    keyword = keyword.strip()
    question_list=''
    c=0
    for question in questions:
        if keyword in question:
            c=c+1
            question_list = question_list + '\n'+ question.strip() +'?'
    # print(question_list)
    # print(c)
    new_row= {'Keyword': keyword,'Question_list':question_list,'No_of_questions':c}
    df = df.append(new_row, ignore_index= True)
df.to_csv('./FLAN_FinBPS/Question_sorted_acc_to_keywords.csv')


