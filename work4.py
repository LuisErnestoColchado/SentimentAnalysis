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

#%%>

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
xTrain = X[0:40000]
xTest = X[40000:50000]
yTrain = Y[0:40000].astype(np.float32)
yTest = Y[40000:50000].astype(np.float32)
#%%
xTrain[0:10]
#%%
numWords = []
for x in xTrain:
    count = len(x)
    numWords.append(count)
mean = np.mean(numWords)
#%%
W2V = Word2Vec(size=int(mean), min_count=10)
W2V.build_vocab(X)
W2V.train(X,total_examples=len(X),epochs=10)

#%%
W2V.most_similar('good')
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
trainDataVecs = getAvgFeatureVecs(xTrain, W2V, int(mean))
testDataVecs = getAvgFeatureVecs(xTest, W2V, int(mean))
#%%
trainDataVecs = trainDataVecs.T
testDataVecs = testDataVecs.T
#%%
#lstmSize = 3
#number_of_layers = 3
#batch_size = 24
#num_steps = 49999
#probabilities = []
#loss = 0.0
#learning_rate = np.power(10.0,-4.0)
#
#W = np.random.rand(1,1)
#b = np.random.rand(1,1)
#
#def lstm_cell(lstm_size):
#  return tf.contrib.rnn.BasicLSTMCell(lstm_size)
#
#def lossFunction(y,currentYTrain):
#    return -1*np.mean(np.multiply(currentYTrain,np.log(y)) + np.multiply((1-currentYTrain),np.log(1-y)))
#
##stacked_lstm = tf.contrib.rnn.MultiRNNCell(
##    [lstm_cell() for _ in range(number_of_layers)])
#
#lstm = lstm_cell(lstmSize)
#state = lstm.zero_state(batch_size, dtype=tf.float32).eval()
#input = np.zeros((batch_size, 100))
#for i in range(0,num_steps):
#    # The value of state is updated after processing each batch of words.
#    #output, state = stacked_lstm(trainDataVecs[:, i], state)
#    input[0:24,:] = trainDataVecs[i:i+24,:]
#    print(input[0:24,:].shape);
#    output, state = lstm(input[0:24,:], state)
#    
#    logits = tf.matmul(output, W) + b
#    probabilities.append(tf.nn.softmax(logits))
#    loss = lossFunction(probabilities, yTrain)
#    optimzer = tf.train.AdadeltaOptimizer (learning_rate).minimize(loss)
#    print("step ",i,"loss",loss)
#final_state = state
#
#def session():
#%%
learning_rate = np.power(10.0,-3.0)
training_steps = 100000
batch_size = 1000
display_step = 1000

num_input = 119
timesteps = 1
num_hidden = 8
num_classes = 1

#%%
# Training Parameters
tf.reset_default_graph()

X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def RNN(x, weights, biases):

    x = tf.unstack(x, timesteps, 1)

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    #print(weights['out'])

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)

prediction = tf.nn.sigmoid(logits)

loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
gvs = optimizer.compute_gradients(loss_op)
capped = [(tf.clip_by_value(grad,-1.,1.),var) for grad,var in gvs]
#train_op = optimizer.minimize(loss_op)
train_op = optimizer.apply_gradients(capped)
optimizer.minimize(loss_op)

p5 = tf.constant(0.5)
delta = tf.abs((Y - prediction))
correct_prediction = tf.cast(tf.less(delta, p5), tf.int32)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
#%%

with tf.Session() as sess:
    accuracyTraining = []
    lossTraining = []
    steps = []
    sess.run(init)
    count = 0
    startData = 0
    endData = batch_size
    
    test_len = 10000
    test_data = testDataVecs[:,:].reshape((-1, timesteps, num_input))
    test_label = yTest[:].reshape((10000,1))
    for step in range(1, training_steps+1):
        #print(step)
        count+=1
        batch_x = trainDataVecs[startData:endData,:]
        batch_y = yTrain[startData:endData]
        #print(count," - ",endData)
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        batch_y = batch_y.reshape((batch_size,1))
        batch_y[:,:].astype(float)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:

            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
            
            lossTraining.append(loss)
            steps.append(step)
            #batch_y = batch_y.
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
        #learning_rate += np.power(10.0,-10.0) 
    print("Optimization Finished!")


#%%
#x = np.linspace(0, 10, 100)
plt.xlabel('step')
plt.ylabel('loss')
plt.plot(steps, lossTraining)
plt.legend()
plt.show()