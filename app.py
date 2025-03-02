# app.py
from sentence_transformers import SentenceTransformer, util
from langdetect import detect
from googletrans import Translator, LANGUAGES
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
translator = Translator()

def translate_if_needed(input_sentence):
    detected_lang = detect(input_sentence)
    if detected_lang != 'en' and detected_lang in LANGUAGES:
        try:
            translated = translator.translate(input_sentence, src=detected_lang, dest='en').text
            return translated
        except Exception as e:
            print(f"Translation failed for {detected_lang}: {e}. Returning original.")
            return input_sentence
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
        "Lyric Snippet": lyrics[most_similar_idx][:100] + "..." if len(lyrics[most_similar_idx]) > 100 else lyrics[most_similar_idx],
        "Similarity Score": f"{cosine_scores[most_similar_idx]:.4f}"
    }
    return "\n".join([f"{key}: {value}" for key, value in output.items()])

interface = gr.Interface(
    fn=find_most_similar_row,
    inputs=gr.Textbox(label="Enter a sentence (any language)"),
    outputs=gr.Textbox(label="Most Similar Song Details"),
    title="Taylor Swift Song Matcher",
    description="Find the most similar Taylor Swift song lyric based on your input, with fast translation."
)

interface.launch()