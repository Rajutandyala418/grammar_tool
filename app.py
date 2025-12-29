from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
from gtts import gTTS
import base64
from io import BytesIO
from deep_translator import GoogleTranslator
import os
from openai import OpenAI

nlp = spacy.load("en_core_web_sm")

OPENAI_KEY = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

app = Flask(__name__)
CORS(app)

POS_MAPPING = {
    "ADJ": "Adjective",
    "ADP": "Adposition (preposition/postposition)",
    "ADV": "Adverb",
    "AUX": "Auxiliary Verb",
    "CONJ": "Conjunction",
    "CCONJ": "Coordinating Conjunction",
    "DET": "Determiner",
    "INTJ": "Interjection",
    "NOUN": "Noun",
    "NUM": "Numeral",
    "PART": "Particle",
    "PRON": "Pronoun",
    "PROPN": "Proper Noun",
    "PUNCT": "Punctuation",
    "SCONJ": "Subordinating Conjunction",
    "SYM": "Symbol",
    "VERB": "Verb",
    "X": "Other"
}

def correct_sentence_gpt(text):
    if not text.strip() or not client:
        return text
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that corrects grammar."},
                {"role": "user", "content": f"Correct the following sentence grammatically. Return only the corrected sentence:\n{text}"}
            ],
        )
        return response.choices[0].message.content.strip()
    except:
        return text

def analyze_pos_tense(text):
    if not text.strip():
        return [], []
    doc = nlp(text)
    pos_tags = [(token.text, POS_MAPPING.get(token.pos_, token.pos_), token.tag_) for token in doc]
    tense_info = []
    for token in doc:
        if token.tag_ in ["VBD", "VBN"]:
            tense_info.append((token.text, "Past"))
        elif token.tag_ in ["VB", "VBP", "VBZ"]:
            tense_info.append((token.text, "Present"))
        elif token.tag_ == "VBG":
            tense_info.append((token.text, "Continuous"))
    return pos_tags, tense_info

def detect_tense(text):
    doc = nlp(text.lower())
    tokens = [t.text for t in doc]
    if "will" in tokens:
        if "have" in tokens and "been" in tokens and any(t.tag_ == "VBG" for t in doc):
            return "Future Perfect Continuous"
        elif "have" in tokens and any(t.tag_ == "VBN" for t in doc):
            return "Future Perfect"
        elif "be" in tokens and any(t.tag_ == "VBG" for t in doc):
            return "Future Continuous"
        else:
            return "Future Simple"
    if any(t.text in ["am", "is", "are"] for t in doc):
        if any(t.tag_ == "VBG" for t in doc):
            return "Present Continuous"
        elif "have" in tokens and "been" in tokens and any(t.tag_ == "VBG" for t in doc):
            return "Present Perfect Continuous"
    if any(t.text in ["has", "have"] for t in doc):
        if "been" in tokens and any(t.tag_ == "VBG" for t in doc):
            return "Present Perfect Continuous"
        elif any(t.tag_ == "VBN" for t in doc):
            return "Present Perfect"
    if "had" in tokens:
        if "been" in tokens and any(t.tag_ == "VBG" for t in doc):
            return "Past Perfect Continuous"
        elif any(t.tag_ == "VBN" for t in doc):
            return "Past Perfect"
    if any(t.text in ["was", "were"] for t in doc):
        if any(t.tag_ == "VBG" for t in doc):
            return "Past Continuous"
    if any(t.tag_ == "VBD" for t in doc):
        return "Past Simple"
    if any(t.tag_ in ["VB", "VBZ"] for t in doc):
        return "Present Simple"
    return "Unknown tense"

@app.route("/process_text", methods=["POST"])
def process_text():
    try:
        data = request.get_json(force=True)
        text = data.get("input_text", "")
        corrected_text = correct_sentence_gpt(text)
        pos_tags, tense_info = analyze_pos_tense(corrected_text)
        detected_tense = detect_tense(corrected_text)
        return jsonify({
            "corrected_text": corrected_text,
            "pos_tags": pos_tags,
            "tense_info": tense_info,
            "detected_tense": detected_tense
        })
    except:
        return jsonify({
            "corrected_text": "Error processing input.",
            "pos_tags": [],
            "tense_info": [],
            "detected_tense": ""
        })

@app.route("/translate_text", methods=["POST"])
def translate_text():
    try:
        data = request.get_json(force=True)
        text = data.get("input_text", "")
        target_language = data.get("target_language", "hi").lower().strip()
        if not text.strip():
            return jsonify({"translated_text": ""})
        translated = GoogleTranslator(source="auto", target=target_language).translate(text)
        return jsonify({"translated_text": translated})
    except Exception as e:
        return jsonify({"translated_text": f"Error: {str(e)}"})

@app.route("/speech_output", methods=["POST"])
def speech_output():
    try:
        data = request.get_json(force=True)
        text = data.get("input_text", "")
        language_name = data.get("language", "English")
        if not text.strip():
            return jsonify({"audio_base64": ""})
        LANG_CODES = {
            "English": "en","Hindi": "hi","Bengali": "bn","Telugu": "te","Marathi": "mr",
            "Tamil": "ta","Urdu": "ur","Gujarati": "gu","Kannada": "kn","Odia": "or",
            "Malayalam": "ml","Punjabi": "pa","Assamese": "as","Sanskrit": "sa"
        }
        lang_code = LANG_CODES.get(language_name, "en")
        tts = gTTS(text=text, lang=lang_code)
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio_base64 = base64.b64encode(mp3_fp.read()).decode("utf-8")
        return jsonify({"audio_base64": audio_base64})
    except:
        return jsonify({"audio_base64": ""})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
