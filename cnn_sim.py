
# coding: utf-8

# In[53]:

# data
data = './data/'
MAX_SEQUENCE_LENGTH = 80
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.3


# In[54]:

import pandas as pd


# In[55]:

data = pd.read_csv('./data/train.csv', na_values='NULL')


# In[56]:

data = data.dropna()
data.head()


# In[57]:

from keras.layers import Input, Conv1D, MaxPool1D, Dense, Activation, Flatten, concatenate, Subtract, Multiply, Embedding


# In[58]:

qus1 = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')
qus2 = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')


# In[59]:

embedding = Embedding(output_dim=EMBEDDING_DIM, input_dim=MAX_NB_WORDS+1, input_length=MAX_SEQUENCE_LENGTH)
x1 = embedding(qus1)
x2 = embedding(qus2)


# In[60]:

conv = Conv1D(32, 4, activation='tanh')
c1 = conv(x1)
c2 = conv(x2)


# In[61]:

maxp = MaxPool1D(pool_size=4)
m1 = maxp(c1)
m2 = maxp(c2)


# In[62]:

fla = Flatten()
f1 = fla(m1)
f2 = fla(m2)


# In[63]:

sub = Subtract()([f1, f2])
mul = Multiply()([f1, f2])
merge = concatenate([sub, mul])


# In[64]:

den1 = Dense(units=20, activation='relu')(merge)


# In[65]:

den2 = Dense(units=1, activation='sigmoid')(den1)


# In[66]:

from keras.models import Model


# In[67]:

model = Model(inputs=[qus1, qus2], outputs=den2)


# In[68]:

data_q1 = data['question1'].values
data_q1.shape


# In[69]:

data_q2 = data['question2'].values
data_q2.shape


# In[70]:

label = data['is_duplicate'].values
label.shape


# In[71]:

data.info()


# In[73]:

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])


# In[74]:

label[0:10]


# In[75]:

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[82]:

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)


# In[84]:

#data_q1, data_q2 = [x.tolist() for x in [data_q1, data_q2]]


# In[85]:

len(data_q1)


# In[86]:

data_q1[0:10]


# In[87]:

tokenizer.fit_on_texts(data_q1+data_q2)


# In[88]:

seq1 = tokenizer.texts_to_sequences(data_q1)
seq2 = tokenizer.texts_to_sequences(data_q2)


# In[96]:

seq1_pad = pad_sequences(seq1, maxlen=MAX_SEQUENCE_LENGTH)


# In[97]:

seq2_pad = pad_sequences(seq2, maxlen=MAX_SEQUENCE_LENGTH)


# In[98]:

seq1_pad[0]


# In[100]:

seq2_pad[0]


# In[102]:

import numpy as np


# In[103]:

print(seq1_pad.shape)
indices = np.arange(seq1_pad.shape[0])


# In[105]:

np.random.shuffle(indices)


# In[107]:

seq1_pad_suf = seq1_pad[indices]
seq2_pad_suf = seq2_pad[indices]


# In[110]:

label_suf = label[indices]


# In[112]:

nb_validation_samples = int(VALIDATION_SPLIT * seq2_pad_suf.shape[0])


# In[113]:

nb_validation_samples


# In[114]:

VALIDATION_SPLIT


# In[118]:

seq1_train = seq1_pad_suf[:-nb_validation_samples]
seq1_test = seq1_pad_suf[-nb_validation_samples:]
seq2_train = seq2_pad_suf[:-nb_validation_samples]
seq2_test = seq2_pad_suf[-nb_validation_samples:]
y_train = label_suf[:-nb_validation_samples]
y_test = label_suf[-nb_validation_samples:]


# In[119]:

seq1_train.shape


# In[120]:

seq1_test.shape


# In[121]:

y_train.shape


# In[122]:

y_test.shape


# In[ ]:

model.fit([seq1_train, seq2_train], y_train, epochs=2)


# In[ ]:



