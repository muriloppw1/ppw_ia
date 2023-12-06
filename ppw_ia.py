import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datetime import datetime
import pytz
import re
import googleapiclient.discovery
import webbrowser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

YOUTUBE_API_KEY = "AIzaSyAJBcghaqNS1YpjGWUrHItwxEoFQChq630"

try:
    import unidecode
except ModuleNotFoundError:
    import unidecode

nltk.download('stopwords')
nltk.download('punkt')

brasil_timezone = pytz.timezone('America/Sao_Paulo')
brasil_locale = 'pt_BR'
stop_words = set(stopwords.words('portuguese'))

qa_pairs = {
    "Quem é PPW_IA?": "Eu sou a PPW.IA, uma I.A. com o intuito de sanar possíveis dúvidas.",
    "voce é mulher": "Eu sou uma I.A, portanto não tenho gênero.",
"Qual é o elemento químico mais abundante na crosta terrestre?": "Oxigênio é o elemento mais abundante na crosta.",
"Qual é o símbolo químico do ouro?": "Au é o símbolo químico do valioso ouro.",
"Quem é a autora de 'Orgulho e Preconceito'?": "Jane Austen é a autora de 'Orgulho e Preconceito'.",
"Qual é a moeda oficial do Japão?": "A moeda oficial do Japão é o iene japonês.",
"Qual é o maior oceano do mundo?": "Oceano Pacífico é o maior oceano do planeta.",
"Quem é o autor de 'O Grande Gatsby'?": "F. Scott Fitzgerald é o autor de 'O Grande Gatsby'.",
"Qual é o país mais populoso do mundo?": "A China é o país mais populoso do globo.",
"Qual é o processo pelo qual as plantas produzem seu próprio alimento?": "A fotossíntese permite que as plantas façam isso.",
"Quem foi o primeiro homem a pisar na Lua?": "Neil Armstrong foi o primeiro na superfície lunar.",
"Qual é o rio mais longo do mundo?": "O Rio Amazonas detém o título de mais longo.",
"Quem é o autor de 'A Origem das Espécies'?": "Charles Darwin é o autor de 'A Origem das Espécies'.",
"Qual é o maior deserto do mundo?": "O Deserto do Saara é o maior deserto global.",
"Qual é a montanha mais alta do mundo?": "O Monte Everest é a montanha mais alta do planeta.",
"Quem é conhecido como o 'pai da informática'?": "Alan Turing é conhecido como o 'pai da informática'.",
"Qual é a fórmula química da água?": "H2O é a fórmula química da água.",
"Qual é a maior ilha do mundo?": "A Groenlândia é a maior ilha do mundo.",
"Quem escreveu 'A Metamorfose'?": "Franz Kafka é o autor de 'A Metamorfose'.",
"Qual é a cidade mais populosa do Brasil?": "São Paulo é a cidade mais populosa do Brasil.",
"Em que ano a Segunda Guerra Mundial terminou?": "A Segunda Guerra Mundial terminou em 1945.",
"Qual é a capital da Austrália?": "Camberra é a capital da Austrália.",
     "Qual país venceu a Copa do Mundo de 1986?": "A Argentina venceu a Copa do Mundo de 1986.",
"Qual país venceu a Copa do Mundo de 1990?": "A Alemanha venceu a Copa do Mundo de 1990.",
}

synonyms = {
    "voce": ["tu", "vc", "sua"],
}

# Criação do modelo de aprendizado de máquina
model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear', probability=True))
questions = list(qa_pairs.keys())
answers = list(qa_pairs.values())
model.fit(questions, answers)

def normalize_text(text):
    text = unidecode.unidecode(text)
    return text.lower()

def find_synonyms(word):
    for key, value in synonyms.items():
        if word in value:
            return key
    return word

def preprocess_text(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

def search_youtube(query):
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    search_response = youtube.search().list(q=query, type="video", part="id", maxResults=1).execute()
    if "items" in search_response:
        video_id = search_response["items"][0]["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        return video_url
    else:
        return "Desculpe, não encontrei nenhum vídeo correspondente."

def get_current_time():
    now = datetime.now(brasil_timezone)
    formatted_time = now.strftime("%A %d/%m/%Y %H:%M")
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

def search_response_in_qa_pairs(normalized_question):
    for q, a in qa_pairs.items():
        if normalize_text(q) in normalized_question:
            return a
    return None

def get_response(question):
    normalized_question = normalize_text(question)

    response_from_qa_pairs = search_response_in_qa_pairs(normalized_question)
    if response_from_qa_pairs:
        return response_from_qa_pairs

    if question.startswith("/musica"):
        query = question[len("/musica"):].strip()
        video_url = search_youtube(query)
        if video_url:
            print("Redirecionando para o vídeo no YouTube...")
            webbrowser.open(video_url)
            return "Aqui está o link para a música que você procurou: " + video_url

    words = normalized_question.split()
    for i, word in enumerate(words):
        synonym = find_synonyms(word)
        words[i] = synonym
    normalized_question = ' '.join(words)

    preprocessed_question = preprocess_text(normalized_question)
    preprocessed_question = ' '.join(preprocessed_question)

    response = model.predict([normalized_question])[0]
    confidence = model.decision_function([normalized_question])[0]

    if confidence > 0.7:
        return response
    else:
        return "Desculpe, não entendi a pergunta."

# Execução do chatbot
while True:
    user_input = input("Usuário: ")
    if user_input.lower() == 'sair':
        break
    response = get_response(user_input)

    print("PPW.IA v1.0:", response)
