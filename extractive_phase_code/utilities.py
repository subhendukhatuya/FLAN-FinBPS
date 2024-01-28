def unique(sent):
    sentences = sent.split('\n')
    unique_list = list(set(sentences))
    s = ''
    for sentence in unique_list:
        s = s + '\n' + sentence
    s = s.strip()
    return s

def clean(string_list):
    d=[]
    for s in string_list:
        if len(s)>0 :
            d.append(s.strip())
    return d

def top_few(a: list[float])->list[int]:
    import numpy as np
    n= len(a)
    m= len(a[0])
    ind= []
    index= [i for i in range(n)]
    for i in range(0,n):
        ind.append(np.mean(a[i]))
    #print(ind)
    for i in range(0,n):
        for j in range(0, n-i-1):
            if(ind[j]<ind[j+1]):
                ind[j],ind[j+1] = ind[j+1],ind[j]
                index[j],index[j+1]= index[j+1],index[j]
    return index[0]


