import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load Hugging Face models
# Sentiment Analysis
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Intent Detection (Text Classification)
intent_classifier = pipeline("text-classification", model="pysentimiento/bert-base-uncased-emo")

# Response Generation
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
response_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Streamlit UI
st.title("Chatbot with Sentiment and Intent Detection")
user_input = st.text_input("Type your message:")

if user_input:
    # Sentiment Analysis
    sentiment = sentiment_analyzer(user_input)[0]
    sentiment_label = sentiment["label"]
    sentiment_score = round(sentiment["score"], 2)

    # Intent Detection
    intent = intent_classifier(user_input)[0]
    intent_label = intent["label"]
    intent_score = round(intent["score"], 2)

    # Response Generation
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    response_ids = response_model.generate(input_ids, max_length=50)
    bot_response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Display Results
    st.write(f"**Sentiment**: {sentiment_label} (Score: {sentiment_score})")
    st.write(f"**Intent**: {intent_label} (Score: {intent_score})")
    st.write(f"**Chatbot Response**: {bot_response}")
