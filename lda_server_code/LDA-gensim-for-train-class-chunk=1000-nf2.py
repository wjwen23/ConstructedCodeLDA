
# coding: utf-8

# # 将class作为文档 进行LDA

# In[ ]:


from tqdm import tqdm
import json

# In[ ]:


import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from lda_model_modify import modify_lda_inference
modify_lda_inference()
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# get_ipython().run_line_magic('matplotlib', 'inline')

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# # 代码相关预处理
# 
# 根据官方文档扩展代码关键词
# https://docs.oracle.com/javase/tutorial/java/nutsandbolts/_keywords.html
# 
# ```json
# ["abstract","continue","for","new","switch","assert","default","goto","package","synchronized","boolean","do","if","private","this","break","double","implements","protected","throw","byte","else","import","public","throws","case","enum","instanceof","return","transient","catch","extends","int","short","try","char","final","interface","static","void","class","finally","long","strictfp","volatile","const","float","native","super","while"]
# ```

# In[ ]:

stop_words = []
code_keywords = ["abstract","continue","for","new","switch","assert","default","goto","package","synchronized","boolean","do","if","private","this","break","double","implements","protected","throw","byte","else","import","public","throws","case","enum","instanceof","return","transient","catch","extends","int","short","try","char","final","interface","static","void","class","finally","long","strictfp","volatile","const","float","native","super","while"]
stop_words.extend(code_keywords)



# 加载类数据
# path_to_token_train= "./total_data/processed_data/tokenized_classes_train.token"
path_to_token_train = "C://Users/Thinkpad/Desktop/tokenresult.txt"

def load_token_data(path_to_token_train):
    data = []
    with open(path_to_token_train, "r") as file:
        lines = file.readlines()
        for line in tqdm(lines):
            # item = json.loads(line)
            # vocabinfo = line.split("+")
            # print(vocabinfo[0])

            data.append(line)
    return data

data = load_token_data(path_to_token_train)


# 这里用到了gensim.utils.simple_preprocess,会自动将太长和太短的词过滤掉, 比如x,y, 比如0,1

# In[ ]:


def sent_to_words(sentences):
    for sentence in tqdm(sentences):
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
data_words = list(sent_to_words(data))
print(data_words[:1])


# In[ ]:
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in tqdm(texts)]
# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)


# In[ ]:

# Create Dictionary
id2word = corpora.Dictionary(data_words_nostops)

# Create Corpus
texts = data_words_nostops

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])
print(len(corpus))

# serialize the corpus
corpora.MmCorpus.serialize('lda_classes_token_train.mm', corpus)

# and reload it!
corpus = gensim.corpora.MmCorpus("lda_classes_token_train.mm")

# In[ ]:


def train_lda_model(dictionary, corpus, texts, limit, start=2, step=3, epoch=5):
    """
    Compute c_v coherence for various number of topics
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    best_model : Best of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    best_value = 0.0
    for num_topics in tqdm(range(start, limit, step)):
        # - emm, do not use multicore...
        #         model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,id2word=id2word,num_topics=num_topics,
        #                                                         random_state=100,chunksize=100,passes=2,
        #                                                         per_word_topics=True,workers=2)
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                   id2word=id2word,
                                                   num_topics=num_topics, 
                                                   random_state=100,
                                                   update_every=1,
                                                   chunksize=1000,
                                                   passes=epoch,
                                                   iterations=2,
                                                   alpha='auto',
                                                   per_word_topics=True)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_value = coherencemodel.get_coherence()
        print("k="+str(num_topics)+", coherence:"+str(coherence_value))
        if coherence_value > best_value:
            print("best model: k="+str(num_topics))
            best_value = coherence_value
            best_model = model
        model.save('lda_models_nf_tokenized/lda_models_'+str(num_topics)+'_epoch='+str(epoch)+'_chunk='+str(1000)+'.model')
        coherence_values.append(coherence_value)
    return best_model, coherence_values


# In[ ]:

limit=150
start=10
step=10
epoch=5

# 训练并获得最优评估值模型
lda_model, coherence_values = train_lda_model(dictionary=id2word, corpus=corpus, texts=data_words_nostops, start=start, limit=limit, step=step, epoch=epoch)

print(coherence_values)

# # # In[ ]:

# # Show graph
# x = range(start, limit, step)
# plt.plot(x, coherence_values)
# plt.xlabel("Num Topics")
# plt.ylabel("Coherence score")
# plt.legend(("coherence_values"), loc='best')
# # plt.show()
# plt.savefig('lda_models_nf_2/topicnum_coherence.png',bbox_inches = 'tight')
