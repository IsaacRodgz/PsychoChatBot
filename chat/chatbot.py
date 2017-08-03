# %%

import os
import json
import nltk
import gensim
import numpy as np
from gensim import corpora, models, similarities
import pickle
from keras.models import Sequential
from keras.layers.recurrent import LSTM, SimpleRNN
from sklearn.model_selection import train_test_split
import tensorflow as tf

# %%

path = "/home/isaac/Documents/Github/PsychoChatBot/chat"
os.chdir(path)

# Load gensim model
model = models.KeyedVectors.load_word2vec_format("SBW-vectors-300-min5.txt")

# Load facebook conversations
data = json.load(open("chats.json"))

# Transform data from json to list
chats = []
for i in range(len(data)):
    chat = []
    data_ = data[str(i)]
    author_1 = data_[0]["author"]
    chat.append(data_[0]["text"])
    for j in range(1, len(data_)):
        if data_[j]["author"] == author_1:
            chat[-1] = chat[-1] + " " + data_[j]["text"]
        else:
            chat.append(data_[j]["text"])
            author_1 = data_[j]["author"]
    chats.append(chat)

chats = chats[:100]
print(len(chats))

# %%

# Build variable X -> Questions, y -> Answers and tokenize them

X = []
y = []

for i in range(len(chats)):
    for j in range(len(chats[i])):
        if j < len(chats[i]) - 1:
            X.append(chats[i][j])
            y.append(chats[i][j + 1])

tok_X = []
tok_y = []
for i in range(len(X)):
    tok_X.append(nltk.word_tokenize(X[i].lower()))
    tok_y.append(nltk.word_tokenize(y[i].lower()))

# %%

sentend = np.ones((300,), dtype=np.float32)

vec_X = []
for sent in tok_X:
    sent_vec = [model[w] for w in sent if w in model.wv.vocab]
    vec_X.append(sent_vec)

vec_y = []
for sent in tok_y:
    sent_vec = [model[w] for w in sent if w in model.wv.vocab]
    vec_y.append(sent_vec)

for tok_sent in vec_X:
    tok_sent[14:] = []
    tok_sent.append(sentend)

for tok_sent in vec_X:
    if len(tok_sent) < 15:
        for i in range(15 - len(tok_sent)):
            tok_sent.append(sentend)

for tok_sent in vec_y:
    tok_sent[14:] = []
    tok_sent.append(sentend)

for tok_sent in vec_y:
    if len(tok_sent) < 15:
        for i in range(15 - len(tok_sent)):
            tok_sent.append(sentend)

with open('conversation.pickle', 'wb') as f:
    pickle.dump([vec_X, vec_y], f)

# %%

with open('conversation.pickle', 'rb') as f:
    vec_X, vec_y = pickle.load(f)

vec_X = np.array(vec_X, dtype=np.float64)
vec_y = np.array(vec_y, dtype=np.float64)

print(vec_X.shape)
print(vec_y.shape)

X_train, X_test, y_train, y_test = train_test_split(vec_X, vec_y,
                                                    test_size=0.2, random_state=1)

model = Sequential()
model.add(LSTM(output_dim=300, input_shape=X_train.shape[1:],
               return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(output_dim=300, input_shape=X_train.shape[1:],
               return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(output_dim=300, input_shape=X_train.shape[1:],
               return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(output_dim=300, input_shape=X_train.shape[1:],
               return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.compile(loss='cosine_proximity', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test))
model.save('LSTM500.h5')
model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test))
model.save('LSTM1000.h5')
model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test))
model.save('LSTM1500.h5')
model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test))
model.save('LSTM2000.h5')
model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test))
model.save('LSTM2500.h5')
model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test))
model.save('LSTM3000.h5')

prediction = model.predict(X_test)
result = [model.most_similar([prediction[10][i]]) for i in range(15)]

# %%
