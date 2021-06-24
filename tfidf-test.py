import math
import pandas as pd
from numba import jit
#import plotly.graph_objects as go
import networkx as nx
data = pd.read_csv("Horizon_Scanning_Sample.csv")
corpus=data["Summary of Source"]
    
#os.environ['MPLCONFIGDIR'] = '/tmp'
#import matplotlib.pyplot as plt

import nltk
#nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')








class Word:
    def __init__(self, word):
        self.word = word
        self.tfidf ={}# ={summary:score}
        
    def u_tfidf(self, score, summary):
        self.tfidf[summary] = score
        
    
    def sort_by_tfidf(self):
        self.tfidf = {key: value for key, value in sorted(self.tfidf.items(), key=lambda item: item[1], reverse=True)}
        
    def to_dict(self):
        dic = {} # {word:summaries}
        dic[self.word] = []
        for Id in self.tfidf:
            dic[self.word].append(Id)
        return dic

#could be useful
class Summary:
    def __init__(self, id, summary):
        self.summary = summary
        self.id = id
        self.words = {}#{word:score}
    
    

            


        


#stop_words.append("would")
#stop_words.append("said")

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
        if len(new_word) <=15:
            
            if new_word[0:3] != "http":
              word_count+=1
              filtered_words.append(new_word)
    sentence_filtered=' '.join(filtered_words)
    return sentence_filtered


def make_corpus_good(corpus):
  new_corpus = []
  #make data nice and stringy
  for documentid in range(len(corpus)):
      try:
      # .strip() doesnt do anything as all data is in a single line .lower().split() are nessesary to make life easier
        new_doc = remove_stopwords(str(corpus[documentid]).strip()).lower().split()
        if len(new_doc)>0:
          new_corpus.append(new_doc)
        # print(str(len(corpus[documentid])) +", ", end='')
      except:
      # check to see which documents break stuff, no longer nessesary
          print(documentid)
  return new_corpus

#returns list of tfidf scores for a given word and list of documents (could be optimised)
#far more efficient as both tf and idf can be calculated simultaneusly
#@jit(nopython=True, cache=True)
def tfidf(word,documents):
    ret = {}
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
    idf = math.log(len(documents)/(fre_of_docs)
    for Id in range(len(scores)):
        ret[Id]=(scores[Id]/length_of_docs[Id])*idf)#converts each score into tfidf score
    return ret

#returns a dictionary of words with the tfidf scores for each document eg the word uk in document 0 would be 0.09...
def calc_tfidf(documents):
    list_of_words=[]
    #generate set of words
    for document in documents:
        for item in document:
            list_of_words.append(Word(item))
    #calculate list of tfidf values for that word
    for word in list_of_words:
        word.tfidf = tfidf(word.word, documents)
    return list_of_words#returns list of objects Word.tfidf = {summary id:score}

#@jit(nopython=False)
def plot_graph(list_of_words,filename):
  
  #create a graph object
  G=nx.Graph()
  #empty atm
  #set of all documents to be displayed
  #for each word in the dictionary
  for word in list_of_words:
      G.add_node(word.word)
      for doc in word.tfidf:
          G.add_node(str(doc))
  for word in list_of_words:
      #add a node to the graph
      #for each document the word is associated with
      for doc in word.tfidf:
          #add the doc to the list_of_docs
          #hopefully this weight is vaguely accurate
          
          G.add_edge(word.word,str(doc),weight = word.tfidf[doc]*1000)
  nx.write_gexf(G, filename) 
  #nx.draw(G, with_labels=True) #Translates graph into visual  
  #plotly_draw_graph(G)

  #plt.show() #Displays visual
        
if __name__ == "__main__":
  #G = nx.random_geometric_graph(200, 0.125)
  #plotly_draw_graph(G)
  new_corpus = make_corpus_good(corpus)

  list_of_words = calc_tfidf(new_corpus)
  '''
  dic = {}
  for word in list_of_words:
      #print(word.tfidf)
      word.sort_by_tfidf()
      #print(word.tfidf, "\n")
      #print(word.to_dict())
      dic.update(word.to_dict())
  
'''
  
  plot_graph(list_of_words, "graph.gexf")
    #dic={"word":[1,4,7,89,456], "another":[1,45,234,667]}
  #data = pd.DataFrame(data = dic)
  #data.to_csv("graph.csv")
  # graph can go straight into the website
