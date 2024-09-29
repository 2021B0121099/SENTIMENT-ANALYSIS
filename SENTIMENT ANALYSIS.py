#!/usr/bin/env python
# coding: utf-8

# In[25]:


get_ipython().system('pip install tensorflow')
get_ipython().system('pip install keras')


# In[26]:


# Importing essential libraries and functions

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from numpy import array
import tensorflow as tf


# In[29]:


# Import from tensorflow.keras instead of keras
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, GlobalMaxPooling1D, Embedding, Conv1D, LSTM
from sklearn.model_selection import train_test_split


# In[31]:


from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[42]:


# Importing dataset
reviews = pd.read_csv("Sentiment Analysis.csv")


# In[43]:


# Dataset exploration
reviews.shape


# In[44]:


reviews.head(5)


# In[45]:


reviews.isnull().values.any()


# In[46]:


# Let's observe distribution of positive / negative sentiments in dataset
import seaborn as sns
sns.countplot(x='sentiment', data=reviews)


# In[47]:



reviews["review"][2]

# You can see that our text contains punctuations, brackets, HTML tags and numbers 
# We will preprocess this text in the next section


# In[54]:


import nltk

# Download the stopwords data
nltk.download('stopwords')


# In[55]:


from b2_preprocessing_function import CustomPreprocess


# In[56]:


custom = CustomPreprocess()
custom.preprocess_text("Those were the best days of my life!")


# In[58]:


# Calling preprocessing_text function on movie_reviews

X = []
sentences = list(reviews['review'])
for sen in sentences:
    X.append(custom.preprocess_text(sen))


# In[60]:


# Sample cleaned up review 

X[2]


# In[62]:


# Converting sentiment labels to 0 & 1
y = reviews['sentiment']
y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))


# In[63]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# The train set will be used to train our deep learning models 
# while test set will be used to evaluate how well our model performs 


# # Preparing embedding layer

# In[64]:


# Embedding layer expects the words to be in numeric form 
# Using Tokenizer function from keras.preprocessing.text library
# Method fit_on_text trains the tokenizer 
# Method texts_to_sequences converts sentences to their numeric form

word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(X_train)

X_train = word_tokenizer.texts_to_sequences(X_train)
X_test = word_tokenizer.texts_to_sequences(X_test)


# In[65]:


import io
import json


# In[66]:


# Saving
tokenizer_json = word_tokenizer.to_json()
with io.open('b3_tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))


# In[67]:


# Adding 1 to store dimensions for words for which no pretrained word embeddings exist

vocab_length = len(word_tokenizer.word_index) + 1

vocab_length


# In[68]:


# Padding all reviews to fixed length 100

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


# In[80]:


# Load GloVe word embeddings and create an Embeddings Dictionary

from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()
glove_file = open('a2_glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()


# In[81]:


# Create Embedding Matrix having 100 columns 
# Containing 100-dimensional GloVe word embeddings for all words in our corpus.

embedding_matrix = zeros((vocab_length, 100))
for word, index in word_tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


# In[82]:


embedding_matrix.shape


# # Simple Neural Network

# In[84]:


from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

# Neural Network architecture
snn_model = Sequential()
embedding_layer = Embedding(vocab_length, 100, weights=[embedding_matrix], trainable=False)

snn_model.add(embedding_layer)
snn_model.add(Flatten())
snn_model.add(Dense(1, activation='sigmoid'))


# In[85]:


# Model compiling

snn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(snn_model.summary())


# In[86]:


# Model training

snn_model_history = snn_model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)


# In[87]:


# Predictions on the Test Set

score = snn_model.evaluate(X_test, y_test, verbose=1)


# In[88]:


# Model Performance

print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# In[89]:


# Model Performance Charts

import matplotlib.pyplot as plt

plt.plot(snn_model_history.history['acc'])
plt.plot(snn_model_history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(snn_model_history.history['loss'])
plt.plot(snn_model_history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()


# # CNN

# In[90]:


from keras.layers import Conv1D


# In[91]:


# Neural Network architecture

cnn_model = Sequential()

embedding_layer = Embedding(vocab_length, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
cnn_model.add(embedding_layer)

cnn_model.add(Conv1D(128, 5, activation='relu'))
cnn_model.add(GlobalMaxPooling1D())
cnn_model.add(Dense(1, activation='sigmoid'))


# In[92]:


# Model compiling

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(cnn_model.summary())


# In[93]:


# Model training

cnn_model_history = cnn_model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)


# In[94]:


# Predictions on the Test Set

score = cnn_model.evaluate(X_test, y_test, verbose=1)


# In[95]:


# Model Performance

print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# In[96]:


# Model Performance Charts

import matplotlib.pyplot as plt

plt.plot(cnn_model_history.history['acc'])
plt.plot(cnn_model_history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()

plt.plot(cnn_model_history.history['loss'])
plt.plot(cnn_model_history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()


# # LSTM

# In[24]:


import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Sample data preparation (replace this with your actual data)
# X should be your input sequences and y should be the corresponding labels
# Ensure your input data is preprocessed appropriately
# X = ...  # Your input sequences
# y = ...  # Your labels

# Example: Generating dummy data for demonstration
vocab_length = 10000  # Size of the vocabulary
maxlen = 100  # Maximum length of input sequences
num_samples = 1000  # Number of samples

# Generating random data for demonstration
X = np.random.randint(1, vocab_length, (num_samples, maxlen))
y = np.random.randint(0, 2, (num_samples, 1))  # Binary labels

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare embedding layer weights (if using pre-trained embeddings)
embedding_matrix = np.random.random((vocab_length, 100))  # Replace with actual embedding weights

# Define the LSTM model
lstm_model = Sequential()
embedding_layer = Embedding(vocab_length, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False)
lstm_model.add(embedding_layer)
lstm_model.add(LSTM(128))
lstm_model.add(Dense(1, activation='sigmoid'))

# Compile the model
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
lstm_model_history = lstm_model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

# Evaluate the model
score = lstm_model.evaluate(X_test, y_test, verbose=1)
print(f"Test Score: {score[0]}")
print(f"Test Accuracy: {score[1]}")

# Plotting model accuracy
plt.plot(lstm_model_history.history['accuracy'])
plt.plot(lstm_model_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plotting model loss
plt.plot(lstm_model_history.history['loss'])
plt.plot(lstm_model_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Save the model
lstm_model.save(f"./c1_lstm_model_acc_{round(score[1], 3)}.keras")


# # Prediction on Live data

# In[28]:


pwd # lists files in working directory


# In[30]:


import keras
print(keras.__version__)


# In[34]:


from keras.models import Sequential
from keras.layers import LSTM, Dense, Input

# Define your input dimensions
timesteps = 5  # Adjust this to your desired number of timesteps
features = 100  # Adjust this to the number of features in your input data

# Define the model architecture
model = Sequential()
model.add(Input(shape=(timesteps, features)))  # Use Input layer
model.add(LSTM(units=128))
model.add(Dense(units=1, activation='sigmoid'))

# Optionally, compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load weights if you have them saved
# model.load_weights(model_path)

model.summary()


# In[35]:


# Load sample IMDb reviews csv, having ~6 movie reviews, along with their IMDb rating

sample_reviews = pd.read_csv("a3_IMDb_Unseen_Reviews.csv")

sample_reviews.head(6)


# In[37]:


# Import the custom module if needed
# import custom  # Uncomment this if you have a separate module

# Example preprocess function if it's not defined in a separate module
def preprocess_text(text):
    # Implement your preprocessing logic here
    return text.lower()  # Example: converting to lowercase

# Preprocess review text with the defined preprocess_text function
unseen_reviews = sample_reviews['Review Text']

unseen_processed = []
for review in unseen_reviews:
    review = preprocess_text(review)  # Call the function directly
    unseen_processed.append(review)

# Now you can use `unseen_processed` for further analysis or predictions


# In[38]:



unseen_processed


# In[39]:


from keras_preprocessing.text import tokenizer_from_json


# In[40]:


# Loading
with open('b3_tokenizer.json') as f:
    data = json.load(f)
    loaded_tokenizer = tokenizer_from_json(data)


# In[41]:


# Tokenising instance with earlier trained tokeniser
unseen_tokenized = loaded_tokenizer.texts_to_sequences(unseen_processed)


# In[42]:


unseen_tokenized


# In[43]:


# Pooling instance to have maxlength of 100 tokens
unseen_padded = pad_sequences(unseen_tokenized, padding='post', maxlen=100)


# In[44]:


unseen_padded


# In[68]:


# Assuming `model` is your trained model
model.save('c1_lstm_model.h5')  # Save again to ensure compatibility


# In[69]:


from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Input

# Define your input dimensions
vocab_size = 10000  # Example vocabulary size
embedding_dim = 128
timesteps = 5  # Your desired number of timesteps

# Define the model architecture
model = Sequential()
model.add(Input(shape=(timesteps,)))  # Use Input layer without input_length
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))  # Embedding layer
model.add(LSTM(units=128))  # LSTM layer
model.add(Dense(units=1, activation='sigmoid'))  # Output layer

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Save the model
model.save('new_model.h5')


# In[70]:


pretrained_lstm_model = load_model('new_model.h5')


# In[73]:


import os
import pandas as pd
import numpy as np
import json
from keras.models import load_model
from keras_preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences

# Specify your model path
model_path = 'new_model.h5'  # Replace with your updated model's file path

# Check if the model file exists
if os.path.exists(model_path):
    # Load your trained LSTM model
    pretrained_lstm_model = load_model(model_path)
else:
    print(f"Model file not found at: {model_path}")
    exit(1)  # Exit if model not found

# Load the tokenizer
with open('b3_tokenizer.json') as f:
    data = json.load(f)
    loaded_tokenizer = tokenizer_from_json(data)

# Define your input dimensions
timesteps = 5  # Adjust this to your desired number of timesteps

# Function to preprocess text
def preprocess_text(text):
    return text.lower()

# Function for live data prediction
def predict_sentiment(review_text):
    processed_review = preprocess_text(review_text)
    tokenized_review = loaded_tokenizer.texts_to_sequences([processed_review])
    padded_review = pad_sequences(tokenized_review, padding='post', maxlen=timesteps)

    # Make predictions (input shape should be (1, timesteps))
    prediction = pretrained_lstm_model.predict(padded_review)
    
    return prediction[0][0]

# Example usage
if __name__ == "__main__":
    review_text = input("Enter a movie review for sentiment prediction: ")
    sentiment = predict_sentiment(review_text)
    sentiment_label = "Positive" if sentiment >= 0.5 else "Negative"
    print(f"The predicted sentiment is: {sentiment_label} (Score: {sentiment})")

