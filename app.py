# Install necessary packages (for requirements.txt)
# pip install transformers torch pandas langdetect sentence-transformers sacremoses gradio

import numpy as np
import pandas as pd
from transformers import pipeline
import torch
from sentence_transformers import SentenceTransformer, util
from langdetect import detect
import gradio as gr

# Load and preprocess the DataFrame once at startup
file_path = "TaylorSwift.csv"  # Update this path for Hugging Face Spaces
df = pd.read_csv(file_path)

# Handle missing values (assuming no 'modes' variable, using a simple approach)
df = df.dropna(subset=['Lyric'])  # Drop rows with missing lyrics
df = df.drop(['Unnamed: 0', 'Artist', 'Year', 'Date'], axis=1, errors='ignore')
df['lyric_length'] = df['Lyric'].str.len()

# Determine the maximum length <= 10000
max_length = df['lyric_length'][df['lyric_length'] <= 10000].max()
print(f"Maximum lyric length <= 10000: {max_length}")

# Filter and sort the DataFrame
df_filtered = df[df['lyric_length'] <= max_length]
df_sorted = df_filtered.sort_values('lyric_length', ascending=False)

# Add summary column (run once at startup)
def add_summary_column(df, batch_size=32):
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=0 if torch.cuda.is_available() else -1
    )
    lyrics = df['Lyric'].tolist()
    summaries = []
    for i in range(0, len(lyrics), batch_size):
        batch = lyrics[i:i + batch_size]
        batch_summaries = summarizer(
            batch,
            max_length=100,
            min_length=30,
            do_sample=False,
            truncation=True
        )
        summaries.extend([result['summary_text'] for result in batch_summaries])
    df['Summary'] = summaries
    return df

# Precompute summaries
df_summary = add_summary_column(df_sorted)

# Function for Gradio
def find_most_similar_row(input_sentence):
    detected_lang = detect(input_sentence)
    if detected_lang != 'en':
        try:
            specific_model = f"Helsinki-NLP/opus-mt-{detected_lang}-en"
            translator = pipeline("translation", model=specific_model, 
                                 device=0 if torch.cuda.is_available() else -1)
            translated = translator(input_sentence, max_length=512)[0]['translation_text']
        except Exception as e:
            translator = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-de", 
                                 device=0 if torch.cuda.is_available() else -1)
            translated = translator(input_sentence, max_length=512)[0]['translation_text']
    else:
        translated = input_sentence

    model = SentenceTransformer('all-MiniLM-L6-v2')
    input_embedding = model.encode(translated, convert_to_tensor=True)
    summaries = df_summary['Summary'].tolist()
    summary_embeddings = model.encode(summaries, convert_to_tensor=True)
    cosine_scores = util.cos_sim(input_embedding, summary_embeddings)[0]
    most_similar_idx = torch.argmax(cosine_scores).item()
    most_similar_row = df_summary.iloc[most_similar_idx]
    
    # Format the output
    output = {
        "Original Input": input_sentence,
        "Translated Input (to English)": translated,
        "Detected Language": detected_lang,
        "Most Similar Song Title": most_similar_row['Title'],
        "Album": most_similar_row['Album'],
        "Summary": most_similar_row['Summary'],
        "Similarity Score": f"{cosine_scores[most_similar_idx].item():.4f}"
    }
    return "\n".join([f"{key}: {value}" for key, value in output.items()])

# Create Gradio interface
interface = gr.Interface(
    fn=find_most_similar_row,
    inputs=gr.Textbox(label="Enter a sentence (any language)"),
    outputs=gr.Textbox(label="Most Similar Song Details"),
    title="Taylor Swift Song Matcher",
    description="Enter a sentence in any language to find the most semantically similar Taylor Swift song based on lyrics summary."
)

# Launch the app
interface.launch()