
Chatbot with Sentiment and Intent Detection
Introduction:
This project is a chatbot application that utilizes Natural Language Processing (NLP) techniques to analyze user input for sentiment classification and intent detection. The chatbot provides real-time conversational responses based on user queries, demonstrating the practical use of pre-trained transformer models. The project is deployed on Streamlit Cloud for easy access via a web browser.

Features:
1. Sentiment Analysis:
  -> Detects whether the user input is positive, negative, or neutral.
  -> Uses the pre-trained DistilBERT model for high accuracy and efficiency.
2. Intent Detection:
  -> Categorizes user input into actionable intents (e.g., Weather Query, Music Request, General Query).
  -> Implements keyword-based logic for intent classification.
3. Chatbot Response Generation:
  -> Generates relevant conversational responses using DialoGPT, a pre-trained transformer model.
4. Web Deployment:
  -> The application is deployed on Streamlit Cloud and accessible via a URL.

Getting Started:
Prerequisites:
  -> Python 3.8+
  -> Install the required dependencies listed in requirements.txt.

Installation:
1. Clone the repository:
  -> git clone https://github.com/Khubaib3737/NLP-Project.git
  -> cd NLP-Project
2. Install dependencies:
  -> pip install -r requirements.txt
3. Run the app locally:
  -> streamlit run app.py
  -> You can also visit: https://nlp-by-sk.streamlit.app/
Access the a;pp in your browser at http://localhost:8501.

Usage:
1. Open the deployed application via the provided Streamlit Cloud URL.
2. Enter a query in the input box (e.g., "What's the weather like today?").
3. View:
  -> Sentiment: Positive, Neutral, or Negative.
  -> Intent: Categorized based on user input (e.g., Weather Query).
  -> Chatbot Response: A conversational reply generated dynamically.\
   
Example Interaction:
Input: "I'm feeling happy today, tell me something fun!"
Output:
Sentiment: Positive
Intent: General Query
Chatbot Response: "I'm glad to hear that! Here's a fun fact: Did you know octopuses have three hearts?"

Technologies Used:
1. Python: Programming language for backend development.
2. Streamlit: Framework for deploying the web application.
3. Hugging Face Transformers:
  -> DistilBERT: For sentiment analysis.
  -> DialoGPT: For chatbot response generation.
4. Torch: Backend for deep learning models.

Deployment:
The application is deployed on Streamlit Cloud:
1. Clone the repository and upload it to GitHub.
2. Connect your GitHub repository to Streamlit Cloud.
3. Deploy the app by selecting app.py as the entry point.

Challenges:
-> Deployment limitations on Streamlit Cloud required optimizing model loading and runtime performance.
-> Simplified intent detection is currently based on keyword matching, which may not handle complex queries.

Future Enhancements:
-> Integrate external APIs (e.g., weather, news, or music) to enhance chatbot functionality.
-> Replace keyword-based intent detection with a fine-tuned intent classification model for better accuracy.
