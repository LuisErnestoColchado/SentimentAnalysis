import numpy as np 
import pandas as pd
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
import tensorflow as tf
import gensim
LabeledSentence = gensim.models.doc2vec.LabeledSentence
from tqdm import tqdm

#%%

# if you want to download the original file:
#df = pd.read_csv('https://raw.githubusercontent.com/rasbt/pattern_classification/master/data/50k_imdb_movie_reviews.csv')
# otherwise load local file
df = pd.read_csv('/home/luisernesto/Documents/MCSII/IntelligentSystem/work4/shuffled_movie_data.csv')
df.tail()

#%%
X = df.loc[:,'review'].values 
Y = df.loc[:,'sentiment'].values

#%%
stop = stopwords.words('english')
porter = PorterStemmer()

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    text = [w for w in text.split() if w not in stop]
    tokenized = [porter.stem(w) for w in text]
    return text

#%%
def tokenizer_start(data):
    for i in range (0,data.shape[0]):
        wordList = tokenizer(data[i])
        data[i] = wordList
#%%
tokenizer_start(X)
#%%
X[0]
#%%
W2V = Word2Vec(size=100, min_count=10)
W2V.build_vocab(X)
W2V.train(X,total_examples=len(X),epochs=10)

#%%
W2V.most_similar('good')
#%%
xTrain = X[0:40000]
xTest = X[40000:50000]
yTrain = Y[0:40000]
yTest = Y[40000:50000]
#%%
xTrain[0:10]
#%%
numWords = []
for x in xTrain:
    count = len(x)
    numWords.append(count)
mean = np.mean(numWords)
#%%
import matplotlib.pyplot as plt
#%matplotlib inline
plt.hist(numWords, 50)
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.axis([0, 1200, 0, 8000])
plt.show()
#%%
# Function to average all word vectors in a paragraph
def featureVecMethod(words, model, num_features):
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0
    
    #Converting Index2Word which is a list to a set for better speed in the execution.
    index2word_set = set(model.wv.index2word)
    
    for word in  words:
        ##print(word)
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    
    # Dividing the result by number of words to get average
    featureVec = np.divide(featureVec, nwords)
    return featureVec

# Function for calculating the average feature vector
def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        # Printing a status message every 1000th review
        if counter%1000 == 0:
            print("Review %d of %d"%(counter,len(reviews)))
            
        reviewFeatureVecs[counter] = featureVecMethod(review, model, num_features)
        counter = counter+1
        
    return reviewFeatureVecs

#%%
trainDataVecs = getAvgFeatureVecs(xTrain, W2V, mean)
testDataVecs = getAvgFeatureVecs(xTest, W2V, mean)
#%%
trainDataVecs = trainDataVecs.T
testDataVecs = testDataVecs.T
#%%
df.shape.ndims
#%%
lstmSize = 3
number_of_layers = 3
batch_size = 24
num_steps = 49999
probabilities = []
loss = 0.0
learning_rate = np.power(10.0,-4.0)

W = np.random.rand(1,1)
b = np.random.rand(1,1)

def lstm_cell(lstm_size):
  return tf.contrib.rnn.BasicLSTMCell(lstm_size)

def lossFunction(y,currentYTrain):
    return -1*np.mean(np.multiply(currentYTrain,np.log(y)) + np.multiply((1-currentYTrain),np.log(1-y)))

#stacked_lstm = tf.contrib.rnn.MultiRNNCell(
#    [lstm_cell() for _ in range(number_of_layers)])

lstm = lstm_cell(lstmSize)
state = lstm.zero_state(batch_size, dtype=tf.float32).eval()
input = np.zeros((batch_size, 100))
for i in range(0,num_steps):
    # The value of state is updated after processing each batch of words.
    #output, state = stacked_lstm(trainDataVecs[:, i], state)
    input[0:24,:] = trainDataVecs[i:i+24,:]
    print(input[0:24,:].shape);
    output, state = lstm(input[0:24,:], state)
    
    logits = tf.matmul(output, W) + b
    probabilities.append(tf.nn.softmax(logits))
    loss = lossFunction(probabilities, yTrain)
    optimzer = tf.train.AdadeltaOptimizer (learning_rate).minimize(loss)
    print("step ",i,"loss",loss)
final_state = state

def session():
    