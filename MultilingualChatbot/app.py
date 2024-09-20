import streamlit as st
import numpy as np
import random
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import nltk
from deep_translator import GoogleTranslator
from langdetect import detect
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
# Chargement du modèle et des données préalablement préparées
model = load_model('my_model.keras')
words, labels, training, output = pickle.load(open("data.pickle", "rb"))

# Charger les intents JSON
import json
with open("intents.json") as file:
    intents = json.load(file)

# Définition de la fonction bag_of_words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

# Initialize Google Translator
translator = GoogleTranslator()

# Streamlit interface
st.title('Chatbot Interface with Translation')

# Session state pour stocker l'historique de la conversation
if 'chat_log' not in st.session_state:
    st.session_state['chat_log'] = []

# Text input pour l'utilisateur
input_text = st.text_input("Vous:")

# Bouton pour envoyer le message
if st.button("Envoyer"):
    # Ajout de la question de l'utilisateur à l'historique
    st.session_state.chat_log.append(("Vous", input_text))
    
    # Détection de la langue de l'entrée
    tokens = nltk.word_tokenize(input_text.lower())
    detected_lang = detect(input_text) 
    # Vérification si le mot "traduire" est présent dans les tokens
    if "translate" in tokens:
        # Traduction du texte si nécessaire
        if detected_lang != 'en':
            try:
                # Extraire le texte à traduire (après "traduire")
                to_translate = input_text.lower().split("translate", 1)[1]
                translated = translator.translate(text=to_translate, source=detected_lang, target='en')
                bot_response = f"Traduction : {translated}"
            except Exception as e:
                bot_response = f"Erreur lors de la traduction: {str(e)}"
        else:
            bot_response = "text in english ."
        st.session_state.chat_log.append(("Bot", bot_response))
    else:
        # Traitement de la réponse du chatbot
        results = model.predict(np.array([bag_of_words(input_text, words)]))
        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in intents["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        # Choix d'une réponse au hasard et ajout à l'historique
        if responses:
            bot_response = random.choice(responses)
        else:
            bot_response = "Je ne peux pas répondre à cela pour le moment."
        st.session_state.chat_log.append(("Bot", bot_response))
    
    # Effacer le champ de saisie après l'envoi du message
    input_text = ""

# Affichage de l'historique de la conversation
for sender, message in st.session_state.chat_log:
    if sender == "Vous":
        st.text("Vous: " + message)
    else:
        st.text("Bot: " + message)
