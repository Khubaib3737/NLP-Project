import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Cache sentiment analysis model
@st.cache_resource
def load_sentiment_model():
    return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Cache chatbot response generation model
@st.cache_resource
def load_response_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    return tokenizer, model

# Load models
sentiment_analyzer = load_sentiment_model()
tokenizer, response_model = load_response_model()

# Streamlit UI
st.title("Chatbot with Sentiment and Intent Detection")
user_input = st.text_input("Type your message:")

if user_input:
    # Sentiment Analysis
    sentiment = sentiment_analyzer(user_input)[0]
    sentiment_label = sentiment["label"]
    sentiment_score = round(sentiment["score"], 2)

    # Simple Intent Detection Logic (Based on Sentiment and Keywords)
    if "weather" in user_input.lower():
        intent_label = "Weather Query"
    elif "music" in user_input.lower():
        intent_label = "Music Request"
    elif sentiment_label == "POSITIVE":
        intent_label = "Positive Statement"
    else:
        intent_label = "General Query"

    # Response Generation
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    response_ids = response_model.generate(input_ids, max_length=50)
    bot_response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Display Results
    st.write(f"**Sentiment**: {sentiment_label} (Score: {sentiment_score})")
    st.write(f"**Intent**: {intent_label}")
    st.write(f"**Chatbot Response**: {bot_response}")
