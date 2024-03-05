"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""

# Visualization dependencies
import plotly.figure_factory as ff
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Code for vectorizer loading
vectorizer_filename = open("resources/vectorizer.pkl","rb")
text_tweets = joblib.load(vectorizer_filename)


# Load your raw data
raw = pd.read_csv("resources/train.csv")




# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit"""

    # Creating sidebar with selection box for model
    model_options = ["Logistic Regression", "Linear Support Vector Classifier", "Support Vector Classifier",
                     "Decision Tree Classifier"]
    selected_model = st.sidebar.selectbox("Choose Model", model_options)

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction","Home","Information"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Home" page
    if selection == "Home":
        image = "resources\imgs\welcomeDD.png"
        st.image(image, use_column_width=True)
        st.write("Unleashing the Power of Dynamic Insights!.")
        home_image_url = "https://news.yale.edu/sites/default/files/styles/card/public/thumbnail/noplanetb.jpg?itok=GiOFmagz&c=455e99c2afc4b0e725221b3110ce48b2"
        st.image(home_image_url, caption="Climate change awareness. Photo source: Yale News", use_column_width=True)
        text = """ 
        Vision Statement:
        To be the leading provider of innovative data-driven solutions, 
        empowering organizations to harness the power of data for informed decision-making and positive societal impact.

        Mission Statement:
        At Data Dynamics, our mission is to leverage cutting-edge technologies and expertise in data analytics to unlock actionable insights from diverse data sources. 
        We are committed to enabling businesses to make informed decisions, driving efficiency, sustainability, and growth. Through our dedication to automation and simplicity,
        we aim to deliver user-friendly solutions that streamline processes and maximize the value of data. With a focus on partnership and collaboration, we strive to contribute
        to a data-driven future that fosters innovation and addresses critical sociental challenges, such as climate change.
                """
        st.write(text)

        # Instructions based on the selected option
        if selection == "Prediction":
            st.write("Explore the impact of climate change through tweet classification.")

        elif selection == "Information":
            st.write("Access valuable information related to climate change sentiment. Review the raw Twitter data and labels, and gain insights into the sentiment class distribution.")
            info_image_url = "https://sancell.com.au/wp-content/uploads/2020/02/carbon-footprint-infographic2-scaled.jpg"
            st.image(info_image_url, caption="Carbon Footprint Infographic", use_column_width=True)

    # Building out the "Information" page
    if selection == "Information":

        # local image file
        image_path = "resources/imgs/info.jpg"

        # Center image
        st.image(image_path, use_column_width=True, output_format='jpg', width=30)

        text = """
         General Information
          
         2 News: the tweet links to factual news about climate change

         1 Pro: the tweet supports the belief of man-made climate change

         0 Neutral: the tweet neither supports nor refutes the belief of man-made climate change

         -1 Anti: the tweet does not believe in man-made climate change Variable definitions
          """

        st.info(text)
        # You can read a markdown file from the supporting resources folder
        

        # Dataframe of sentiments and tweets
        st.subheader("Raw Twitter data and label")
        st.subheader("Sentiments:")

        # Buttons
        selected_sentiment1 = st.button("News = 2")
        selected_sentiment2 = st.button("Pro = 1")
        selected_sentiment3 = st.button("Neutral = 0")
        selected_sentiment4 = st.button("Anti = -1")
        selected_sentiment5 = st.button("All")
 
        # Button function
        if selected_sentiment5:
            st.write(raw[['sentiment', 'message']])

        if selected_sentiment1:
            filtered_data1 = raw[raw['sentiment'] == 2]
            st.write(filtered_data1[['sentiment', 'message']])

        if selected_sentiment2:
            filtered_data2 = raw[raw['sentiment'] == 1]
            st.write(filtered_data2[['sentiment', 'message']])

        if selected_sentiment3:
            filtered_data3 = raw[raw['sentiment'] == 0]
            st.write(filtered_data3[['sentiment', 'message']])

        if selected_sentiment4:
            filtered_data4 = raw[raw['sentiment'] == -1]
            st.write(filtered_data4[['sentiment', 'message']])

    # Building out the "Prediction" page
    if selection == "Prediction":

        image = "resources\imgs\predictions.jpg"
        st.image(image, use_column_width=True)
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text", "")

        if st.button("Classify"):
            try:
                # Transforming user input with the appropriate vectorizer
                if selected_model == "Logistic Regression":
                    vect_text = tweet_cv.transform([tweet_text]).toarray()
                else:
                    vect_text = text_tweets.transform([tweet_text]).toarray()  # Use your own vectorized data for other models

                # Load the selected model and get model information
                if selected_model == "Logistic Regression":
                    model_description = "Logistic Regression Model"
                    model_info = "This model is based on logistic regression and is trained to classify tweets into sentiment categories."
                    predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"), "rb"))
                elif selected_model == "Linear Support Vector Classifier":
                    model_description = "Linear Support Vector Classifier Model"
                    model_info = "This model is based on linear support vector classification and is trained to classify tweets into sentiment categories."
                    predictor = joblib.load(open(os.path.join("resources/lsvc_model.pkl"), "rb"))
                elif selected_model == "Support Vector Classifier":
                    model_description = "Support Vector Classifier Model"
                    model_info = "This model is based on support vector classification and is trained to classify tweets into sentiment categories."
                    predictor = joblib.load(open(os.path.join("resources/svc_model.pkl"), "rb"))
                elif selected_model == "Decision Tree Classifier":
                    model_description = "Decision Tree Classifier Model"
                    model_info = "This model is based on the decision tree classification and is trained to classify tweets into sentiment categories."
                    predictor = joblib.load(open(os.path.join("resources/dtc_model.pkl"), "rb"))
                else:
                    st.error("Invalid model selected.")

                # Make predictions
                prediction = predictor.predict(vect_text)

                # Mapping numerical predictions to sentiment labels
                sentiment_labels = {
                   -1: "Anti",
                    0: "Neutral",
                    1: "Pro",
                    2: "News"
                }

                # Displaying the sentiment class description
                sentiment_description = sentiment_labels.get(prediction[0], "Unknown")
                st.success(f"Text Categorized as: {sentiment_description}")

                # Display model information
                st.subheader("Model Information")
                st.write(f"Selected Model: {model_description}")
                st.write(f"Description: {model_info}")

                # Display Word Cloud
                st.subheader("Word Cloud")
                wordcloud = WordCloud(width=800, height=400, max_words=150, background_color='white').generate(tweet_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot()

                # Display Bar Chart
                st.subheader("Predicted Sentiment Distribution")
                prediction_counts = pd.Series(prediction).map(sentiment_labels).value_counts()
                st.bar_chart(prediction_counts)

            except Exception as e:
                st.error(f"An error occurred: {e}")

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()