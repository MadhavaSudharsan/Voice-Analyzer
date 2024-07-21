from flask import Flask, request, jsonify, render_template
from googletrans import Translator
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from collections import Counter
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
import os
import logging
import tempfile
import subprocess

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load configurations from environment variables or a config file
DATABASE_URL = os.getenv('DATABASE_URL', 'mysql+mysqlconnector://root:Madhav_05@localhost/voice_analyzer')

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

# Download NLTK data and load SpaCy model
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

# Define language code mapping
language_name_to_code = {
    'afrikaans': 'af', 'albanian': 'sq', 'arabic': 'ar', 'armenian': 'hy',
    'bengali': 'bn', 'bosnian': 'bs', 'catalan': 'ca', 'croatian': 'hr',
    'czech': 'cs', 'danish': 'da', 'dutch': 'nl', 'english': 'en',
    'esperanto': 'eo', 'estonian': 'et', 'filipino': 'tl', 'finnish': 'fi',
    'french': 'fr', 'german': 'de', 'greek': 'el', 'gujarati': 'gu',
    'hindi': 'hi', 'hungarian': 'hu', 'icelandic': 'is', 'indonesian': 'id',
    'italian': 'it', 'japanese': 'ja', 'javanese': 'jw', 'kannada': 'kn',
    'khmer': 'km', 'korean': 'ko', 'latin': 'la', 'latvian': 'lv',
    'lithuanian': 'lt', 'macedonian': 'mk', 'malayalam': 'ml', 'marathi': 'mr',
    'myanmar': 'my', 'nepali': 'ne', 'norwegian': 'no', 'polish': 'pl',
    'portuguese': 'pt', 'punjabi': 'pa', 'romanian': 'ro', 'russian': 'ru',
    'serbian': 'sr', 'sinhala': 'si', 'slovak': 'sk', 'slovenian': 'sl',
    'spanish': 'es', 'sundanese': 'su', 'swahili': 'sw', 'swedish': 'sv',
    'tamil': 'ta', 'telugu': 'te', 'thai': 'th', 'turkish': 'tr',
    'ukrainian': 'uk', 'urdu': 'ur', 'vietnamese': 'vi', 'welsh': 'cy',
    'xhosa': 'xh', 'yiddish': 'yi', 'yoruba': 'yo', 'zulu': 'zu'
}

# Define database models
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100))
    email = Column(String(100))

class Transcription(Base):
    __tablename__ = 'transcriptions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    content = Column(Text)

class WordFrequency(Base):
    __tablename__ = 'word_frequencies'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    word = Column(String(100))
    frequency = Column(Integer)

Base.metadata.create_all(engine)

def most_frequent_words(text):
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.isalnum()]
    freq_dist = Counter(words)
    return freq_dist.most_common()

def unique_phrases(text, n=3):
    doc = nlp(text)
    phrases = [chunk.text for chunk in doc.noun_chunks]
    return phrases[:n]

def similarity_detector(user_transcriptions, all_transcriptions):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([user_transcriptions] + all_transcriptions)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[:-4:-1]
    return similar_indices[1:].tolist()  # Exclude the first index as it is the user's own transcription

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    temp_audio_path = None
    converted_audio_path = None
    
    try:
        audio = request.files['file']
        user_id = request.form.get('user_id')

        # Check if user exists
        user = session.query(User).filter_by(id=user_id).first()
        if not user:
            return jsonify({'error': 'User ID does not exist.'}), 400

        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_audio_file:
            audio.save(temp_audio_file.name)
            temp_audio_path = temp_audio_file.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
            converted_audio_path = temp_wav_file.name

        # Convert the audio file to a format that speech_recognition can process
        conversion_command = [
            'ffmpeg', '-i', temp_audio_path, '-ac', '1', '-ar', '16000', '-y', converted_audio_path
        ]
        subprocess.run(conversion_command, check=True)

        recognizer = sr.Recognizer()
        with sr.AudioFile(converted_audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        user_transcription = Transcription(user_id=user_id, content=text)
        session.add(user_transcription)
        session.commit()

        word_freq = most_frequent_words(text)
        for word, freq in word_freq:
            word_frequency = WordFrequency(user_id=user_id, word=word, frequency=freq)
            session.add(word_frequency)
        session.commit()

        unique_phrases_list = unique_phrases(text)
        all_transcriptions = [trans.content for trans in session.query(Transcription).all()]
        similar_users = similarity_detector(text, all_transcriptions)

        return jsonify({
            'transcription': text,
            'most_frequent_words': word_freq,
            'unique_phrases': unique_phrases_list,
            'similar_users': similar_users
        })

    except sr.RequestError as e:
        logging.error(f"Could not request results from Google Speech Recognition service; {e}")
        return jsonify({'error': 'Speech recognition request failed.'}), 500
    except sr.UnknownValueError:
        logging.error("Google Speech Recognition could not understand audio")
        return jsonify({'error': 'Speech recognition could not understand audio.'}), 400
    except Exception as e:
        logging.error(f"An error occurred during transcription: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary files
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if converted_audio_path and os.path.exists(converted_audio_path):
            os.remove(converted_audio_path)

@app.route('/translate', methods=['POST'])
def translate():
    try:
        text = request.form['text']
        translator = Translator()
        translated_text = translator.translate(text, dest='en').text
        return jsonify({'translated_text': translated_text})
    except Exception as e:
        logging.error(f"An error occurred during translation: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        logging.error(f"Error starting the Flask application: {e}")









