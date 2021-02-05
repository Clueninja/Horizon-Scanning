import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
import time

import networkx as nx
import matplotlib.pyplot as plt



class PLSA_modeller:
    def __init__(self,csv_filename,desired_column_title,num_topics):
        self.df = pd.read_csv(csv_filename)
        self.data_summaries = self.df[[desired_column_title]].astype('str')
        self.desired_column_title = desired_column_title
        self.filtered_column_title = 'filtered desired field'
        self.num_topics=num_topics



    def filter(self):
        ##filtering out stopwords
        self.stop_words = stopwords.words('english')
        self.data_summaries[self.filtered_column_title] = self.data_summaries[self.desired_column_title].apply(lambda x: self.remove_stopwords(x))

        ##start making matrices out of these
        self.summary_sentences = [''.join(sentence) for sentence in self.data_summaries[self.filtered_column_title]]
        return self.summary_sentences

    def remove_stopwords(self,sentence_unfiltered):
        words = sentence_unfiltered.split(' ')
        self.word_count=0
        words = [w for w in words if w not in self.stop_words]
        filtered_words=[]
        for word in words:
            word=word.lower()
            new_word=''
            for char_index in range(len(word)):
                if ord(word[char_index]) >= 97 and ord(word[char_index]) <= 122:
                    new_word=new_word+word[char_index]
            if len(new_word) <=10:
                self.word_count+=1
                filtered_words.append(new_word)
        self.sentence_filtered=' '.join(filtered_words)
        return self.sentence_filtered


    def modelling(self):

        ## TODO : learn what sklean countvectoriser does
        self.vectoriser = CountVectorizer(analyzer='word', max_features=5000)
        self.fitted_vector = self.vectoriser.fit_transform(self.summary_sentences)

        ## TODO : learn what sklean TfidfTransformer does
        self.transformer = TfidfTransformer(smooth_idf=False)
        self.fitted_transform = self.transformer.fit_transform(self.fitted_vector)

        ## TODO : learn what sklean normalize does
        self.tfidf_norm = normalize(self.fitted_transform, norm='l1', axis=1)

        ## TODO : learn what sklean NMF model does
        # number of topics:
        # NMF model (whatever that is):
        self.model = NMF(n_components=self.num_topics, init='nndsvd',max_iter=500)
        self.model.fit(self.tfidf_norm)

    def get_model_info(self):
        ##how many sentences you want to retrieve:
        self.n_words_wanted=int(self.df.shape[0])

        #self.n_words_wanted=100

        ## TODO : learn what this is retrieving
        feature_names=self.vectoriser.get_feature_names()

        word_dict={}
        for i in range(self.num_topics):
            words_ids=self.model.components_[i].argsort()[:-self.n_words_wanted - 1:-1]
            words=[feature_names[key] for key in words_ids]
            word_dict['Topic # '+'{:02d}'.format(i+1)] = words


        self.words_to_topics_dataframe=pd.DataFrame(word_dict)
        return self.words_to_topics_dataframe, self.n_words_wanted




class DecisionTreeRegressor_class:
    def __init__(self,y_train,x_train,x_test,num_sentences):
        self.y_train = y_train
        self.x_train = x_train
        self.x_test = x_test
        self.num_sentences = num_sentences

    def train(self):
        self.random_tree_regressor = RandomForestRegressor(n_estimators=5)
        self.random_tree_regressor.fit(self.x_train,self.y_train)

    def test(self):
        self.predictions = self.random_tree_regressor.predict(self.x_test)
        return self.predictions

    def error(self,y_test):
        y_test_list = y_test.tolist()
        y_test_index_list=list(y_test.index)
        absolute_error = 0
        for element in range(len(y_test_list)):
            if y_test_list[element]!=self.predictions[element]:
                absolute_error+=1
        return str(100*absolute_error/len(y_test_list))+'%',y_test_index_list


def GetKey(val,dictA):
    for list_a in list(dictA.items()):
        if val in list_a[1]:
           return True,list_a[0]
    return False, -1

if __name__ == '__main__':

    start_time_modelling = time.time()
    topics=50

    horizon_scanner = PLSA_modeller('Horizon_Scanning_Sample.csv','Summary of Source',topics)
    sentences = horizon_scanner.filter()
    horizon_scanner.modelling()
    PLSA_results, num_of_sentences=horizon_scanner.get_model_info()
    end_time_modelling = time.time()
    print('PLSA complete in',end_time_modelling-start_time_modelling,'seconds')

    start_time_trees=time.time()

    y_unfiltered = horizon_scanner.df['Category'].apply(lambda x :x[:2])
    y_length=len(y_unfiltered)
    y_unfiltered_train = y_unfiltered[[x for x in range(y_length) if y_unfiltered[x]!='12']]
    y_train=y_unfiltered_train.apply(lambda x: int(x.strip('.')))

    y_unfiltered_unknowns = y_unfiltered[[x for x in range(y_length) if y_unfiltered[x]=='12']]
    y_unknowns=y_unfiltered_unknowns.apply(lambda x: int(x.strip('.')))
    indexes_of_y=list(y_unknowns.index)


    y_unfiltered_list = y_unfiltered.tolist()
    indexes_test = []
    for element_idx in range(len(y_unfiltered_list)):
        if y_unfiltered_list[element_idx]=='12':
            indexes_test.append(element_idx)

    ##sorting out x - matrix where rows are 0-1306 and columns are the topics
    #fetch the topics:
    topics_found=[]
    topic_tracker={}

    for sentence in range(num_of_sentences):
        for topic in range(topics):
            if PLSA_results.iloc[sentence,topic] in topics_found:
                topic_tracker[PLSA_results.iloc[sentence,topic]].append(sentence)
            else:
                topics_found.append(PLSA_results.iloc[sentence,topic])
                topic_tracker[PLSA_results.iloc[sentence, topic]]=[sentence]


    dictionary_for_dataframe_x = {}
    for topic in topic_tracker:
        list_to_add = []
        prev_index=-1
        dupes_gone = list(dict.fromkeys(topic_tracker[topic]))
        for new_index in dupes_gone:
            index = new_index-prev_index
            for i in range(index-1):
                list_to_add.append(0)
            list_to_add.append(1)
            prev_index=new_index
        for numbers_left in range(num_of_sentences-1-new_index):
            list_to_add.append(0)
        dictionary_for_dataframe_x[topic]=list_to_add


    x=pd.DataFrame(dictionary_for_dataframe_x)



    x_train = x.iloc[[i for i in range(y_length) if i not in indexes_test]]
    x_test = x.iloc[[i for i in range(y_length) if i in indexes_test],:]


    treeeeee = DecisionTreeRegressor_class(y_train,x_train,x_test,num_of_sentences)


    treeeeee.train()
    preds = treeeeee.test()
    #percentage_error, indexes_of_y = treeeeee.error()
    end_time_trees=time.time()
    print('Forest complete in',end_time_trees-start_time_trees,'seconds')
    #print('percentage error is:',percentage_error)




    ##graphing
    g = nx.Graph()

    ##format predictions so for each prediction is a list of indexes
    indexes_of_preds={}
    preds_found=[]

    total_connections=100

    for pred in range(len(preds)):
        if str(preds[pred]) in preds_found:
            indexes_of_preds[str(preds[pred])].append(indexes_of_y[pred])
        else:
            preds_found.append(preds[pred])
            indexes_of_preds[str(preds[pred])]=[indexes_of_y[pred]]

    ##format dictionary of predictions so each prediction becomes a tuple within a list
    connections=[]
    dict_count=0
    for prediction in indexes_of_preds:
        for item in indexes_of_preds[prediction]:
            connections.append((int(float(prediction))*10000,item))
            dict_count+=1
    for i in range(total_connections-dict_count):
        y_train_list = list(y_train)
        y_train_indexes = list(y_train.index)
        connections.append((int(y_train_list[i])*10000,y_train_indexes[i]))
    g.add_edges_from(connections)

    color_map = []
    colours=['#E31025','#E31088','#CE10E3','#4A10C7','#1075C7','#10AFC7','#10C793','#10C756','#10C713','#C1C710','#FF7D03']

    for node in g:
        present, category = GetKey(node, indexes_of_preds)
        try:
            category = y_train_list[y_train_indexes.index(node)]
            present=True
        except:
            pass
        if present:
            color_map.append(colours[int(float(category))-1])
        else:
            color_map.append('#000000')

    pos = nx.spring_layout(g)

    nx.draw(g,pos, node_color=color_map, with_labels=True,node_size=20,font_size=7)

    plt.savefig("filename.png")
    plt.show()
    


###credits to https://towardsdatascience.com/topic-modelling-with-plsa-728b92043f41
    #for their article on a PLSA model

###credits to github user DhruvilKarani for his repository explaining how to produce a PLSA model
    #link: https://github.com/DhruviKarani/Non-Negative-Matrix-Factorisation