# Next Word Prediction

Welcome to the Next Word Prediction project! This project leverages advanced machine learning techniques to predict the next word in a given sequence of text. The system is built using TensorFlow and Keras for deep learning, and Streamlit for a user-friendly web interface.

## 📂 Project Overview

This project aims to build a next-word prediction model using Long Short-Term Memory (LSTM) networks. The system includes a complete pipeline from data preparation and model training to a web interface for user interaction.

## 🔍 Architecture

### 1. **Data Preparation**
   - **Data Source**: Text data is loaded from `data/data.txt`.
   - **Preprocessing**: Text is tokenized and converted into sequences. Padding is applied to ensure consistent input size for the model.

### 2. **Model Training**
   - **Model Type**: LSTM-based neural network.
   - **Layers**:
     - **Embedding Layer**: Converts word indices to dense vectors.
     - **LSTM Layers**: Multiple LSTM layers to capture sequential dependencies.
     - **Dense Layer**: Output layer with softmax activation to predict the next word.
   - **Training**: The model is trained on tokenized sequences with appropriate loss functions and optimizers.

### 3. **Prediction**
   - **Input Processing**: User input is tokenized and padded.
   - **Prediction**: The model predicts the next word based on the input sequence.
   - **Output**: The most probable word is selected and displayed.

### 4. **Web Interface**
   - **Framework**: Streamlit.
   - **Features**:
     - Input text area for user input.
     - Button to trigger the prediction process.
     - Display area for showing predicted next words.

## 🛠️ Installation

To set up the project on your local machine, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/Next_Word_Prediction.git
   cd Next_Word_Prediction
