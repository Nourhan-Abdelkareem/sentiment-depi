import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud



# Load your trained model
model = pickle.load(open('D:\Data science\Final Project\data\svm_model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('D:\Data science\Final Project\data\TFIDF_model.pkl', 'rb'))  # Load vectorizer

# Page configuration
st.set_page_config(page_title="Sentiment Analysis App", layout="wide", initial_sidebar_state="expanded")

# Sidebar information
with st.sidebar:
    st.write("### Model Info:")
    st.write("Sentiment Analysis Model v1.0")
    st.write("Accuracy: 93%")
    st.write("Type of Model: SVM")  # Replace with your model type

# Title and description
st.title("Sentiment Analysis")
st.write("Analyze text or upload a file to predict sentiment.")

# Text input for real-time feedback
user_input = st.text_area("Enter text for sentiment analysis:", value="Type here...")

if st.button("Analyze Sentiment"):
    if user_input and user_input != "Type here...":
        # Preprocess the text using the vectorizer
        user_input_tfidf = tfidf_vectorizer.transform([user_input])

        # Make the prediction
        prediction = model.predict(user_input_tfidf)

        # Display sentiment with visualization
        if prediction[0] == "Positive":
            st.success("😊 Positive")
        elif prediction[0] == "Negative":
            st.error("😠 Negative")
        else:
            st.warning("😐 Neutral") 

# File upload for batch sentiment analysis
uploaded_file = st.file_uploader("Upload a CSV file for batch sentiment analysis", type=["csv"])
st.write('Data must have a text column')

if uploaded_file is not None:
    # Read the file into a DataFrame
    data = pd.read_csv(uploaded_file)
    # Expected column names
    expected_columns = ['text']
    
    if set(expected_columns).issubset(data.columns):
        st.write("The uploaded file has the correct column!")
        st.write(data.head())
        
        # Make predictions
        # Transform the text using the TF-IDF vectorizer
        text_data_tfidf = tfidf_vectorizer.transform(data['text'])

        # Predict sentiments for the text data
        predictions = model.predict(text_data_tfidf)

        # Add predictions to the DataFrame
        data['Sentiment'] = predictions 
        
        st.write("### Sentiment Analysis Results:")
        st.write(data)

        # Sentiment distribution chart
        sentiment_counts = data['Sentiment'].value_counts()
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', ax=ax, color=['green', 'red', 'gray'])
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

        # Generate a word cloud
        text_data = ' '.join(data['text'])  # Combine all text
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
        # Display the word cloud
        st.image(wordcloud.to_array(), use_column_width=True, caption='Word Cloud of Text Data')

        # Summary Section
        positive_count = sum(data['Sentiment'] == "Positive")
        neutral_count = sum(data['Sentiment'] == "Neutral")
        negative_count = sum(data['Sentiment'] == "Negative")

        st.write("### Summary:")
        st.write(f"Total texts analyzed: {len(data)}")
        st.write(f"Positive: {positive_count}")
        st.write(f"Neutral: {neutral_count}")
        st.write(f"Negative: {negative_count}")
    else:
        st.error(f"The uploaded file doesn't have the correct columns. Expected: {expected_columns}, but got: {data.columns.tolist()}")

# User feedback section
st.write("### Feedback")
feedback = st.text_input("How accurate was the prediction? (1-5)")
if st.button("Submit Feedback"):
    st.write("Thank you for your feedback!")

# Footer or final notes
st.write("App built with Streamlit.")

#streamlit run d:/Data science/Final Project/data/sentiment_app.py