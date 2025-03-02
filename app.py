# app.py
from sentence_transformers import SentenceTransformer, util
from langdetect import detect
from deep_translator import GoogleTranslator
import gradio as gr
import pickle
import torch

# Load the fine-tuned model and metadata
model = SentenceTransformer('./taylor_swift_finetuned_model')
with open('taylor_swift_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
lyrics = metadata['lyrics']
titles = metadata['titles']
albums = metadata['albums']
lyric_embeddings = torch.tensor(metadata['lyric_embeddings'])

# Initialize the translator
translator = GoogleTranslator(source='auto', target='en')

def translate_if_needed(input_sentence):
    detected_lang = detect(input_sentence)
    print(f"Detected language: {detected_lang}")  # Debug: Log detected language
    
    if detected_lang != 'en':
        try:
            translated = translator.translate(input_sentence)
            if translated == input_sentence or translated is None:  # Check for failed translation
                print(f"Warning: Translation failed or returned identical text for '{input_sentence}' (lang: {detected_lang})")
                return input_sentence
            print(f"Translated '{input_sentence}' to '{translated}'")
            return translated
        except Exception as e:
            print(f"Translation error for '{input_sentence}' (lang: {detected_lang}): {e}")
            return input_sentence  # Fallback to original on error
    return input_sentence

def find_most_similar_row(input_sentence):
    translated = translate_if_needed(input_sentence)
    input_embedding = model.encode(translated, convert_to_tensor=True)
    cosine_scores = util.cos_sim(input_embedding, lyric_embeddings)[0]
    most_similar_idx = torch.argmax(cosine_scores).item()
    
    output = {
        "Original Input": input_sentence,
        "Translated Input (to English)": translated,
        "Detected Language": detect(input_sentence),
        "Most Similar Song Title": titles[most_similar_idx],
        "Album": albums[most_similar_idx],
        "Full Lyric": lyrics[most_similar_idx],  # Return full lyric instead of snippet
        "Similarity Score": f"{cosine_scores[most_similar_idx]:.4f}"
    }
    return "\n".join([f"{key}: {value}" for key, value in output.items()])

interface = gr.Interface(
    fn=find_most_similar_row,
    inputs=gr.Textbox(label="Enter a sentence (any language)"),
    outputs=gr.Textbox(label="Most Similar Song Details", lines=20),  # Increased lines for full lyrics
    title="Taylor Swift Song Matcher",
    description="Find the most similar Taylor Swift song lyric based on your input, with fast translation."
)

interface.launch()