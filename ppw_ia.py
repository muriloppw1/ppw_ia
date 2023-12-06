import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datetime import datetime
import pytz
import re
import difflib
import googleapiclient.discovery
import webbrowser
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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
    # Adicione mais pares de pergunta-resposta conforme necessário
}

synonyms = {
    "voce": ["tu", "vc", "sua"],
}

# Modelo GPT-2 pré-treinado
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

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

def generate_gpt_response(prompt, max_length=50):
    input_ids = gpt_tokenizer.encode(prompt, return_tensors="pt")
    output_ids = gpt_model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    response = gpt_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

def get_response(question):
    normalized_question = normalize_text(question)

    response_from_qa_pairs = qa_pairs.get(normalized_question)
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

    best_match = max(qa_pairs.keys(), key=lambda q: difflib.SequenceMatcher(None, normalized_question, normalize_text(q)).ratio())
    similarity_score = difflib.SequenceMatcher(None, normalized_question, normalize_text(best_match)).ratio()
    similarity_threshold = 0.71
    if similarity_score >= similarity_threshold:
        return qa_pairs[best_match]

    ml_response = generate_gpt_response(normalized_question, max_length=100)
    return ml_response

while True:
    user_input = input("Usuário: ")
    if user_input.lower() == 'sair':
        break

    response = get_response(user_input)

    if response == "Desculpe, não entendi a pergunta.":
        response = "Desculpe, não entendi a pergunta."

    print("PPW.IA:", response)
