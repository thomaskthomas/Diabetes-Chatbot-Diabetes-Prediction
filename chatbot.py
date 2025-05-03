import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import nltk
nltk.download('punkt')
nltk.download('stopwords')


# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load chatbot training data
with open("train_data.json", encoding="utf-8") as file:
    intents = json.load(file)

# Load NLP components
words = pickle.load(open("words.pkl", "rb"))  # Vocabulary list
classes = pickle.load(open("classes.pkl", "rb"))  # Categories list
model = load_model("chatbot_model.h5")  # Trained neural network

def clean_sentence(sentence):
    """Tokenize and lemmatize user input."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """Convert sentence into BoW vector."""
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1  # Mark words that appear in the sentence
    return np.array(bag)

def predict_intent(user_input):
    """Predict intent from user input."""
    bow = bag_of_words(user_input)
    res = model.predict(np.array([bow]))[0]  # Predict intent probabilities
    threshold = 0.25  # Confidence threshold
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)

    return classes[results[0][0]] if results else "fallback"

def get_response(user_input):
    """Generate response based on predicted intent."""
    intent = predict_intent(user_input)
    
    for intent_data in intents["intents"]:
        if intent_data["tag"] == intent:
            return random.choice(intent_data["responses"])  # Return a random response from that intent
    
    return "I'm not sure how to answer that. Can you rephrase?"

# Testing the chatbot
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye! Take care.")
            break
        response = get_response(user_input)
        print(f"Chatbot: {response}")
