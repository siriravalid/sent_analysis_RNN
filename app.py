import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import nltk
nltk.download('punkt')
import pandas as pd
df = pd.read_csv('data.csv')
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'].values)
max_len = 100
X = tokenizer.texts_to_sequences(df['text'].values)
X = pad_sequences(X, maxlen=max_len)
y = df['label'].values
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=128, input_length=max_len),
    tf.keras.layers.SimpleRNN(128),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)
model.save('sentiment_rnn_model.h5')
model = tf.keras.models.load_model('sentiment_rnn_model.h5')
st.title('Sentiment Analysis Tool- Siri')
st.write('Enter text to analyze its sentiment.')
text_input = st.text_area("Enter text here:", height=100)
if st.button('Analyze'):
    if text_input:
        sequence = tokenizer.texts_to_sequences([text_input])
        padded_sequence = pad_sequences(sequence, maxlen=max_len)
        prediction = model.predict(padded_sequence)[0][0]
        if prediction > 0.5:
            st.write("Positive Sentiment")
        else:
            st.write("Negative Sentiment")
    else:
        st.write("Please enter some text to analyze.")
