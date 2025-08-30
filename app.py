from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
from gtts import gTTS
import base64
from io import BytesIO
from deep_translator import GoogleTranslator

# Load SpaCy English model for NLP
nlp = spacy.load("en_core_web_sm")

# Initialize OpenAI client only if available and key provided
import os
OPENAI_KEY = os.environ.get("OPENAI_KEY", "").strip()
client = OpenAI(api_key=OPENAI_KEY) if (OPENAI_AVAILABLE and OPENAI_KEY) else None

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

POS_MAPPING = {
    "ADJ": "Adjective", "ADP": "Adposition (preposition/postposition)", "ADV": "Adverb",
    "AUX": "Auxiliary Verb", "CONJ": "Conjunction", "CCONJ": "Coordinating Conjunction",
    "DET": "Determiner", "INTJ": "Interjection", "NOUN": "Noun", "NUM": "Numeral",
    "PART": "Particle", "PRON": "Pronoun", "PROPN": "Proper Noun", "PUNCT": "Punctuation",
    "SCONJ": "Subordinating Conjunction", "SYM": "Symbol", "VERB": "Verb", "X": "Other"
}

GENDER_KEYWORDS = {
    "boy": "male", "man": "male", "male": "male",
    "girl": "female", "woman": "female", "female": "female"
}

PRONOUNS = {
    "male": {"subject": "he", "object": "him", "possessive": "his"},
    "female": {"subject": "she", "object": "her", "possessive": "her"}
}

def detect_gender(text):
    text_lower = text.lower()
    for word, gender in GENDER_KEYWORDS.items():
        if word in text_lower:
            return gender
    return None

def correct_pronouns(text):
    gender = detect_gender(text)
    if not gender:
        return text
    replacements = {
        "he": PRONOUNS[gender]["subject"], "she": PRONOUNS[gender]["subject"],
        "him": PRONOUNS[gender]["object"], "her": PRONOUNS[gender]["object"],
        "his": PRONOUNS[gender]["possessive"]
    }
    words = text.split()
    corrected_words = [replacements.get(w.lower(), w) for w in words]
    return " ".join(corrected_words)

# ✅ Grammar correction using OpenAI only if available
def correct_sentence_gpt(text):
    if not text.strip():
        return ""
    text = correct_pronouns(text)

    if not client:  # Skip OpenAI if not configured
        return text  

    prompt = f'Correct the following sentence grammatically: "{text}" Return only the corrected sentence.'
    try:
        response = client.responses.create(model="gpt-4o-mini", input=prompt, temperature=0)
        return response.output_text or text
    except Exception as e:
        print("OpenAI API Error:", e)
        return text

# ✅ POS & Tense Analysis with SpaCy
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

@app.route("/process_text", methods=["POST"])
def process_text():
    try:
        data = request.get_json(force=True)
        text = data.get("input_text", "")
        corrected_text = correct_sentence_gpt(text)
        pos_tags, tense_info = analyze_pos_tense(corrected_text)
        return jsonify({"corrected_text": corrected_text, "pos_tags": pos_tags, "tense_info": tense_info})
    except Exception as e:
        print("Error in /process_text:", e)
        return jsonify({"corrected_text": "Error processing input.", "pos_tags": [], "tense_info": []})
@app.route("/translate_text", methods=["POST"])
def translate_text():
    try:
        data = request.get_json(force=True)
        text = data.get("input_text", "")
        target_language = data.get("target_language", "hi").lower().strip()  # ✅ enforce lowercase code

        if not text.strip():
            return jsonify({"translated_text": ""})

        translated = GoogleTranslator(source="auto", target=target_language).translate(text)
        return jsonify({"translated_text": translated})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"translated_text": f"Error: {str(e)}"})



# ✅ Speech Output with gTTS
@app.route("/speech_output", methods=["POST"])
def speech_output():
    try:
        data = request.get_json(force=True)
        text = data.get("input_text", "")
        language_name = data.get("language", "English")
        if not text.strip():
            return jsonify({"audio_base64": ""})

        LANG_CODES = {
            "English": "en", "Hindi": "hi", "Bengali": "bn", "Telugu": "te",
            "Marathi": "mr", "Tamil": "ta", "Urdu": "ur", "Gujarati": "gu",
            "Kannada": "kn", "Odia": "or", "Malayalam": "ml", "Punjabi": "pa",
            "Assamese": "as", "Sanskrit": "sa"
        }
        lang_code = LANG_CODES.get(language_name, "en")

        tts = gTTS(text=text, lang=lang_code)
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio_base64 = base64.b64encode(mp3_fp.read()).decode("utf-8")
        return jsonify({"audio_base64": audio_base64})

    except Exception as e:
        print("Error in /speech_output:", e)
        return jsonify({"audio_base64": ""})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

# ✅ Test OpenAI API Key
@app.route("/test_openai", methods=["GET"])
def test_openai():
    if not client:
        return jsonify({"status": "❌ OpenAI not configured or missing key"})
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input="Say 'Hello from GrammarTool!'"
        )
        return jsonify({"status": "✅ OpenAI working", "reply": response.output_text})
    except Exception as e:
        return jsonify({"status": "❌ OpenAI error", "error": str(e)})


