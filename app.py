# !pip install sentence-transformers torch langdetect gradio

from sentence_transformers import SentenceTransformer, util
from langdetect import detect
import gradio as gr
import pickle
import torch

model = SentenceTransformer('./taylor_swift_finetuned_model')
with open('taylor_swift_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
summaries = metadata['summaries']
titles = metadata['titles']
albums = metadata['albums']
summary_embeddings = torch.tensor(metadata['summary_embeddings'])

def translate_if_needed(input_sentence):
    detected_lang = detect(input_sentence)
    if detected_lang != 'en':
        print(f"Non-English detected ({detected_lang}), but translation not implemented.")
        return input_sentence
    return input_sentence

def find_most_similar_row(input_sentence):
    translated = translate_if_needed(input_sentence)
    input_embedding = model.encode(translated, convert_to_tensor=True)
    cosine_scores = util.cos_sim(input_embedding, summary_embeddings)[0]
    most_similar_idx = torch.argmax(cosine_scores).item()
    
    output = {
        "Original Input": input_sentence,
        "Translated Input (to English)": translated,
        "Detected Language": detect(input_sentence),
        "Most Similar Song Title": titles[most_similar_idx],
        "Album": albums[most_similar_idx],
        "Summary": summaries[most_similar_idx],
        "Similarity Score": f"{cosine_scores[most_similar_idx]:.4f}"
    }
    return "\n".join([f"{key}: {value}" for key, value in output.items()])

interface = gr.Interface(
    fn=find_most_similar_row,
    inputs=gr.Textbox(label="Enter a sentence (any language)"),
    outputs=gr.Textbox(label="Most Similar Song Details"),
    title="Taylor Swift Song Matcher",
    description="Find the most similar Taylor Swift song using a fine-tuned sentence-transformers model."
)

interface.launch()