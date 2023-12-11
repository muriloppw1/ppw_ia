import numpy as np
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sympy import sympify
from sklearn.pipeline import make_pipeline
import difflib
import googleapiclient.discovery
import webbrowser
from datetime import datetime
import pytz
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import unidecode
import string
import re


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/words.zip')
except LookupError:
    nltk.download('words')

try:
    nltk.data.find('corpora/rslp')
except LookupError:
    nltk.download('rslp')
    
from nltk.stem import RSLPStemmer
YOUTUBE_API_KEY = "AIzaSyAJBcghaqNS1YpjGWUrHItwxEoFQChq630"
brasil_timezone = pytz.timezone('America/Sao_Paulo')
stop_words = set(stopwords.words('portuguese'))

qa_pairs = {
}

synonyms = {
   
}

nb_model = make_pipeline(TfidfVectorizer(), MultinomialNB())
questions = list(qa_pairs.keys())
answers = list(qa_pairs.values())
nb_model.fit(questions, answers)
stemmer = RSLPStemmer()
rf_model = make_pipeline(TfidfVectorizer(), RandomForestClassifier())
rf_model.fit(questions, answers)
def normalize_text(text):
    text = unidecode.unidecode(text)
    return text.lower()
def find_synonyms(word):
    for key, value in synonyms.items():
        if word in value:
            return key
    return word

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens if word.lower() not in stop_words]
    return stemmed_tokens

def search_youtube(query):
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    search_response = youtube.search().list(q=query, type="video", part="id", maxResults=1).execute()
    if "items" in search_response:
        video_id = search_response["items"][0]["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        return video_url
    else:
        return "Desculpe, não encontrei nenhum vídeo correspondente."

import calendar
calendario = {
    'Sunday': 'Domingo',
    'Monday': 'Segunda-feira',
    'Tuesday': 'Terça-feira',
    'Wednesday': 'Quarta-feira',
    'Thursday': 'Quinta-feira',
    'Friday': 'Sexta-feira',
    'Saturday': 'Sábado'
}

def get_current_time():
    now = datetime.now(brasil_timezone)
    day_of_week_english = calendar.day_name[now.weekday()]
    day_of_week_portuguese = calendario.get(day_of_week_english, day_of_week_english)
    formatted_time = now.strftime(f"{day_of_week_portuguese} %d/%m/%Y %H:%M")
    return formatted_time
    
def get_creation_date():
    creation_date = datetime(2023, 9, 11, tzinfo=brasil_timezone)
    today = datetime.now(brasil_timezone)
    age = today - creation_date
    return "Fui criado em 11/09/2023 pelo [DEV] MURILOPPW o que me faz ter " + str(age.days) + " dias."

def get_release_date():
    return "Data de lançamento: 11/09/2023"

def get_current_day():
    now = datetime.now(brasil_timezone)
    day_of_week = now.strftime("%A")
    return "Hoje é " + day_of_week

def calculate_similarity(question, reference):
    return difflib.SequenceMatcher(None, question, reference).ratio()

def get_ml_response(question):
    nb_prediction = nb_model.predict([question])[0]
    rf_prediction = rf_model.predict([question])[0]
    if max(nb_model.predict_proba([question])[0]) > max(rf_model.predict_proba([question])[0]):
        return nb_prediction
    else:
        return rf_prediction
        
def calculate_math(expression):
    try:
        result = sympify(expression)
        formatted_result = int(result) if result.is_integer else result
        return f"A resposta é {formatted_result}"
    except Exception as e:
        return f"Ocorreu um erro ao calcular a expressão: {str(e)}"

def get_response(question):
    normalized_question = normalize_text(question)

    match = re.search(r'\d+\s*[+\-*/]\s*\d+(\s*[+\-*/]\s*\d+)*', normalized_question)
    if match:
        expression = match.group(0)
        return calculate_math(expression)
    if question.startswith("/musica"):
        query = question[len("/musica"):].strip()
        video_url = search_youtube(query)
        if video_url:
            print("Redirecionando para o vídeo no YouTube...")
            webbrowser.open(video_url)
            return "Aqui está o link para a música que você procurou: " + video_url
    numbers_and_operators = re.findall(r'\d+\s*[+\-*/]\s*\d+(\s*[+\-*/]\s*\d+)*', normalized_question)
    if numbers_and_operators:
        return calculate_math(numbers_and_operators[0])
    numbers_and_operators_only = re.findall(r'\d+|[+\-*/]', normalized_question)
    reconstructed_expression = ' '.join(numbers_and_operators_only)
    if re.match(r'\d+\s*[+\-*/]\s*\d+(\s*[+\-*/]\s*\d+)*', reconstructed_expression):
        return calculate_math(reconstructed_expression)
    preprocessed_question = preprocess_text(normalized_question)
    preprocessed_question = ' '.join(preprocessed_question)
    for q, a in qa_pairs.items():
        if normalize_text(q) in normalized_question:
            if a == "get_current_time":
                return get_current_time()
            elif a == "get_creation_date":
                return get_creation_date()
            elif a == "get_release_date":
                return get_release_date()
            elif a == "get_current_day":
                return get_current_day()
            else:
                return a
    best_match = max(qa_pairs.keys(), key=lambda q: calculate_similarity(normalized_question, normalize_text(q)))
    similarity_score = calculate_similarity(normalized_question, normalize_text(best_match))
    similarity_threshold = 0.71
    if similarity_score >= similarity_threshold:
        return qa_pairs[best_match]
    return "Desculpe, não entendi a pergunta."
    ml_response = get_ml_response(normalized_question)
    return ml_response
    
while True:
    user_input = input("Usuário: ")
    if user_input.lower() == 'sair':
        break
    response = get_response(user_input)

    if response == "Desculpe, não entendi a pergunta.":
        response = "Desculpe, não entendi a pergunta."

    print("PPW.IA v1.0:", response)
