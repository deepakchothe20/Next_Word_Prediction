from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# # Example corpus (replace with your actual text data)
with open("data.txt", "r", encoding="utf-8") as file:
    data = file.readlines() 

# Create and fit the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)

# Load the pre-trained model
model = tf.keras.models.load_model('next_word_model.h5')

def predict_next_words(text, model, tokenizer, num_words=10):
    """
    Predict the next words given an initial text using the provided model and tokenizer.

    Parameters:
    - text (str): The initial text for prediction.
    - model (tf.keras.Model): The trained LSTM model for prediction.
    - tokenizer (Tokenizer): The tokenizer used for text tokenization.
    - num_words (int): The number of words to predict and append.

    Returns:
    - str: The updated text with predicted words appended.
    """
    for i in range(num_words):
        token_text = tokenizer.texts_to_sequences([text])[0]
        padded_token_text = pad_sequences([token_text], maxlen=32, padding='pre')
        # Predict the next word
        pos = np.argmax(model.predict(padded_token_text))

        # Find the word corresponding to the predicted index
        for word, index in tokenizer.word_index.items():
            if index == pos:
                text = text +" " + word
                print(text)
                break
    return text

# Streamlit application
def main():
    st.title("Next Word Predictor")
    st.write("Enter a starting text and get the next words predicted by the model:")

    input_text = st.text_area("Input Text", "")

    if st.button("Predict Next Words"):
        if input_text:
            # Run the prediction
            result = predict_next_words(input_text, model, tokenizer, num_words=10)
            print(result)
            st.write("Predicted text:", result)
        else:
            st.write("Please enter some text.")
            
    if st.button("Show Data"):
        st.write("Here is the data being used for tokenization:")
        with open("data.txt", "r", encoding="utf-8") as file:
            data_content = file.read()
        st.text_area("Data Content", data_content, height=300)

if __name__ == "__main__":
    main()