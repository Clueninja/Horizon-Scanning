import math
import pandas as pd
data = pd.read_csv("Horizon_Scanning_Sample.csv")
corpus=data["Summary"]
    

import nltk
stop_words = nltk.corpus.stopwords.words('english')


'''
lines that may need removing:
2369
2269
2213
1987
'''
  
# initializing punctuations string  
punc = r'''!()-{}[];:''”"“’\,<>./?@#$%^&*_~'''
  
# Removing punctuations in string 
def rem_punc(string):
    for ele in string:  
        if ele in punc:  
            string = string.replace(ele, "") 
    return string


def remove_stopwords(sentence_unfiltered):
    words = sentence_unfiltered.split(' ')
    word_count=0
    words = [w for w in words if w not in stop_words]
    filtered_words=[]
    for word in words:
        word=word.lower()
        new_word=''
        for char_index in range(len(word)):
            if ord(word[char_index]) >= 97 and ord(word[char_index]) <= 122:
                new_word=new_word+word[char_index]
        if len(new_word) <=10:
            word_count+=1
            filtered_words.append(new_word)
    sentence_filtered=' '.join(filtered_words)
    return sentence_filtered

#make data nice and stringy
for documentid in range(len(corpus)):
    try:
    # .strip() doesnt do anything as all data is in a single line .lower().split() are nessesary to make life easier
        corpus[documentid]=rem_punc(str(corpus[documentid]).strip()).lower().split()
       # print(str(len(corpus[documentid])) +", ", end='')
    except:
    # check to see which documents break stuff, no longer nessesary
        print(documentid)




#long hand inefficient way of doing the same thing
def tf(word, document):
   n=0
   for item in document:
       if word == item:
           n+=1
   return n/len(document)

def idf(word,documents):
    n=0
    for document in documents:
        for item in document:
            if word == item:
                n+=1
    return math.log(len(documents)/(n+1))+1

#returns list of tfidf scores for a given word and list of documents (could be optimised)
#far more efficient as both tf and idf can be calculated simultaneusly
def tfidf(word,documents):
    scores=[]#stores the frequency of a word in a document in a list eg scores=[0,1,0,0,2] if there are 5 documents
    length_of_docs=[]#stores length of each document 
    fre_of_docs=0#stores frequency of the documents that contain the word
    fre_of_word=0#stores total frequency of the word in all documents =sum(scores) before tfidf calcultated
    docid=0#so we can increment the correct document score
    for document in documents:
        length_of_docs.append(len(document))
        scores.append(0)
        found = False# to figure out if the document contains the word, if it does increment num
        for item in document:
            if word == item:
                scores[docid]+=1# increments the correct value in scores so that each document has its own count of the number of times the word appears
                fre_of_word+=1
                if not found:
                    fre_of_docs+=1
                    found = True
        docid+=1
    for Id in range(len(scores)):
        scores[Id] = (scores[Id]/length_of_docs[Id])*(math.log(len(documents)/(fre_of_docs+1)+1))#converts each score into tfidf score
    return scores

#returns a dictionary of words with the tfidf scores for each document eg the word uk in document 0 would be 0.09...
def calc_tfidf(documents):
    list_of_words={}
    #generate set of words
    for document in documents:
        for item in document:
            list_of_words[item]=[]
    #calculate list of tfidf values for that word
    for key in list_of_words:
        list_of_words[key] = tfidf(key, documents)
    return list_of_words




list_of_words = calc_tfidf(corpus)
dicts={}
for w, s  in list_of_words.items():
    doc_links = []
    for i in range(len(s)):
        if s[i]>0.01: #checking if the words exists
            doc_links.append(i+1) #i is increased by 1 to reflect the actual doc number
    dicts[w]=doc_links#list of document numbers that are linked by the words; could be used as seen fit




#import a library that helps with networks
import networkx as nx
import matplotlib.pyplot as plt
#create a graph object
G=nx.Graph()
#empty atm
#set of all documents to be displayed
#for each word in the dictionary
for word in dicts:
    for doc in dicts[word]:
        G.add_node(str(doc))
for word in dicts:
    #add a node to the graph
    G.add_node(word)
    #for each document the word is associated with
    for doc in dicts[word]:
        #add the doc to the list_of_docs
        G.add_edge(word,str(doc))

nx.draw(G, with_labels=True) #Translates graph into visual
plt.show() #Displays visual
