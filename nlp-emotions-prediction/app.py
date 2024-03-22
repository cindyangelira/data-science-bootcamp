import os
import pandas as pd
import streamlit as st
import plotly.express as px
from transformers import pipeline
from google.colab import userdata
from huggingface_hub import login

# Load config
MODEL_ID = os.getenv("MODEL_ID", "cindyangelira/distilbert-base-uncased-finetuned-emotion")
APP_TITLE = os.getenv("APP_TITLE", "Emotions Predictions")
INPUT_PROMPT = os.getenv("INPUT_PROMPT", "Enter your text here")
PLOT_TITLE = os.getenv("PLOT_TITLE", 'Class probability (%)')

# Define the mapping
LABEL_TO_EMOTION = {
    'LABEL_0': 'sadness',
    'LABEL_1': 'joy',
    'LABEL_2': 'love',
    'LABEL_3': 'anger',
    'LABEL_4': 'fear',
    'LABEL_5': 'surprise'
}

st.title(APP_TITLE)

# get model from hub
classifier = pipeline("text-classification", model=MODEL_ID)

# create widget for input text
custom_tweet = st.text_input(INPUT_PROMPT)

if custom_tweet:
    preds = classifier(custom_tweet, top_k = None)
    preds_df = pd.DataFrame(preds)
    st.write(f'The prediction is {preds}')
    preds_df['label'] = preds_df['label'].map(LABEL_TO_EMOTION)
    st.write(f'The prediction is {preds_df.sort_values("label", ascending = False)["label"][0]}')

    # Plot
    fig = px.bar(preds_df, x="label", y="score", title=f'"{custom_tweet}"', labels={'score':PLOT_TITLE}, height=400)
    st.plotly_chart(fig)
