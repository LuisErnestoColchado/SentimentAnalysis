#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import matplotlib.pyplot as plt


# ## Reading data

# In[2]:


df = pd.read_csv('/home/luisernesto/Documents/MCSII/IntelligentSystem/work4/shuffled_movie_data.csv')
df.tail()


# ## Preparing data

# In[3]:


X = df.loc[:,'review'].values 
Y = df.loc[:,'sentiment'].values


# In[4]:


stop = stopwords.words('english')
porter = PorterStemmer()

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    text = [w for w in text.split() if w not in stop]
    tokenized = [porter.stem(w) for w in text]
    return text


# In[5]:


def tokenizer_start(data):
    for i in range (0,data.shape[0]):
        wordList = tokenizer(data[i])
        data[i] = wordList


# ### tokenizing reviews 

# In[6]:


tokenizer_start(X)


# In[7]:


X[0]


# ### Train and test Data 

# In[8]:


xTrain = X[0:40000]
xTest = X[40000:50000]
yTrain = Y[0:40000].astype(np.float32)
yTest = Y[40000:50000].astype(np.float32)


# ### Embedding reviews 

# We calculate the average number of words in all the reviews

# In[9]:


numWords = []
for x in xTrain:
    count = len(x)
    numWords.append(count)
mean = np.mean(numWords)


# In[10]:


W2V = Word2Vec(size=int(mean), min_count=10)
W2V.build_vocab(X)
W2V.train(X,total_examples=len(X),epochs=10)


# In[11]:


W2V.most_similar('good')


# ### Histogram of number words in the reviews 

# In[12]:


plt.hist(numWords, 50)
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.axis([0, 1200, 0, 8000])
plt.show()


# ### Calculate the average of words in a review 

# In[13]:


# Function to calculate the average of all word vectors in a review
def featureVecMethod(words, model, num_features):
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0
    
    #Converting Index2Word which is a list to a set for better speed in the execution.
    index2word_set = set(model.wv.index2word)
    
    for word in  words:
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


# In[14]:


print("Train data")
trainDataVecs = getAvgFeatureVecs(xTrain, W2V, int(mean))
print("Test data")
testDataVecs = getAvgFeatureVecs(xTest, W2V, int(mean))


# ## LTSM and GRU with TensorFlow 

# ### Parameters 

# In[28]:


learning_rate = np.power(10.0,-2.0)
training_steps = 20000
batch_size = 1000
display_step = 1000
num_input = 119
timesteps = 1
num_units = 119
num_classes = 1
num_layers = 9


# ### Implement bidirectional and multilayer LTSM RNN

# In[29]:


tf.reset_default_graph()

X = tf.placeholder("float", [None, timesteps,num_input])
Y = tf.placeholder("float", [None,num_classes])

keep_prob = 1.0

weights = {
    'out': tf.Variable(tf.random_normal([num_units, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}
initializer = tf.random_uniform_initializer(-1, 1)
def RNN(x, weights, biases):
    outputs = x
    #track through the layers
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer),reuse=tf.AUTO_REUSE):
            
            #forward cells
            cell_fw = tf.contrib.rnn.LSTMCell(num_units,initializer=initializer)
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob = keep_prob)
            #backward cells
            cell_bw = tf.contrib.rnn.LSTMCell(num_units,initializer=initializer)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob = keep_prob)

            (output_fw, output_bw), last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw, outputs,dtype=tf.float32)
            state = last_state
            
    rnn_outputs_fw = tf.reshape(output_fw, [-1, num_units])
    rnn_outputs_bw = tf.reshape(output_bw, [-1, num_units])
    out_fw = tf.matmul(rnn_outputs_fw, weights['out']) + biases['out']
    out_bw = tf.matmul(rnn_outputs_bw, weights['out']) + biases['out']
    return np.add(out_fw,out_bw)

logits = RNN(X, weights, biases)

prediction = tf.nn.sigmoid(logits)

loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
#calculate gradients 
gvs = optimizer.compute_gradients(loss_op)

#clipping gradients 
def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1, 1)
capped = [(ClipIfNotNone(grad),var) for grad,var in gvs]
train_op = optimizer.apply_gradients(capped)
optimizer.minimize(loss_op)

p5 = tf.constant(0.5)
delta = tf.abs((Y - prediction))
correct_prediction = tf.cast(tf.less(delta, p5), tf.int32)

#calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()


# ## Start session 

# In[30]:


accuracyTraining = []
lossTraining = []
steps = []
def startSession():
    with tf.Session() as sess:
        sess.run(init)
        count = 0
        startData = 0
        endData = batch_size
        
        test_len = 10000
        test_data = testDataVecs[:,:].reshape((-1, timesteps, num_input))
        test_label = yTest[:].reshape((10000,1))
        
        for step in range(1, training_steps+1):
            count+=1
            batch_x = trainDataVecs[startData:endData,:]
            batch_y = yTrain[startData:endData]
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            batch_y = batch_y.reshape((batch_size,1))
            batch_y[:,:].astype(float)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            
            if step % display_step == 0 or step == 1:
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " +                       "{:.4f}".format(loss) + ", Training Accuracy= " +                       "{:.3f}".format(acc))

                lossTraining.append(loss)
                steps.append(step)
                accu = sess.run(accuracy, feed_dict={X: test_data, Y: test_label})
                print("Testing Accuracy: ", accu)
                accuracyTraining.append(accu)
            if(endData<trainDataVecs.shape[0]):
                startData=count*batch_size
                endData = (count+1)*batch_size
            else:
                count=0
                startData = 0
                endData = batch_size
        print("Optimization Finished!")


# ### Results 

# In[31]:


print("**********LTSM***********")
startSession()
ltsm_loss_training = lossTraining
ltsm_accuracy = accuracyTraining


# In[33]:


plt.xlabel('Step')
plt.ylabel('Testing Accuracy')
plt.plot(steps, ltsm_accuracy,label = 'Accuracy LSTM')
plt.legend()
plt.show()


# In[34]:


plt.xlabel('Step')
plt.ylabel('Training Loss')
plt.plot(steps, ltsm_loss_training,label = 'loss LSTM')
plt.legend()
plt.show()


# In[38]:


print("Accuracy: ",ltsm_accuracy[int(training_steps/display_step)])

