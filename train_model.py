import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import time

#
with open("data.txt", "r", encoding="utf-8") as file:
    data = file.read()



# Tokenize the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
for line in data.split("."):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[: i + 1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
)

# Create predictors and label
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

print(X.shape, y.shape)

model = Sequential()
model.add(Embedding(493, 100, input_length=32))
model.add(LSTM(150))
model.add(Dense(493, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
model.fit(X, y, epochs=100)
model.save("next_word_model.h5")


text = "The future"


for i in range(10):
    # tokenize
    token_text = tokenizer.texts_to_sequences([text])[0]
    # padding
    padded_token_text = pad_sequences([token_text], maxlen=32, padding="pre")
    # predict
    pos = np.argmax(model.predict(padded_token_text))

    for word, index in tokenizer.word_index.items():
        if index == pos:
            text = text + " " + word
            print(text)
            time.sleep(2)
