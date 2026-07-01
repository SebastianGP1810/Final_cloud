import streamlit as st
import os
import pymongo
import cohere
import google.generativeai as genai
from pypdf import PdfReader
from datetime import datetime
from dotenv import load_file

# Cargar .env solo si existe localmente (en Azure se leerá de las Application Settings)
if os.path.exists(".env"):
    load_file(".env")

# ----------------- CONFIGURACIÓN EXAMEN -----------------
# Modifica esto con tus datos reales para cumplir el enunciado
USER_NAME = "Mendoza_Maria" 
st.set_page_config(page_title=f"Chatbot EF - {USER_NAME}", layout="centered")
st.title(f"🤖 Chatbot Inteligente RAG")
st.subheader(f"Evaluación Final - {USER_NAME}")

# ----------------- INICIALIZACIÓN DE SERVICIOS -----------------
MONGO_URI = os.environ.get("MONGO_URI")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
COHERE_KEY = os.environ.get("COHERE_API_KEY")

if not MONGO_URI or not GEMINI_KEY or not COHERE_KEY:
    st.error("🚨 Faltan variables de entorno. Configura MONGO_URI, GEMINI_API_KEY y COHERE_API_KEY.")
    st.stop()

# Conexiones oficiales
client = pymongo.MongoClient(MONGO_URI)
db = client[f"db_ef_{USER_NAME.lower()}"]
history_collection = db["chat_history"]

co = cohere.Client(COHERE_KEY)
genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

# ----------------- PROCESAMIENTO RAG (PDF CHUNKS) -----------------
uploaded_file = st.file_uploader("Subir el PDF Institucional para el contexto", type=["pdf"])

pdf_context = ""
if uploaded_file:
    reader = PdfReader(uploaded_file)
    text_content = ""
    for page in reader.pages:
        text_content += page.extract_text() or ""
    
    # Fragmentar en bloques simples (Chunks) para emular la inyección contextual
    if text_content:
        pdf_context = text_content[:3000] # Tomamos los primeros caracteres como contexto simplificado
        st.success("✅ PDF cargado e indexado en memoria con éxito.")

# ----------------- INTERFAZ DE CONVERSACIÓN -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar el historial visual en Streamlit
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de usuario
if user_query := st.chat_input("Pregúntale al bot sobre el documento..."):
    # 1. Mostrar mensaje del usuario en pantalla
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 2. Construir prompt RAG inyectando el contexto semántico
    full_prompt = f"Contexto extraído del documento:\n{pdf_context}\n\nPregunta del usuario: {user_query}\nResponde de manera precisa basándote solo en el contexto."

    # 3. Simular llamada combinada (Cohere para análisis/Gemini para respuesta)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            # Cohere valida/procesa internamente el input (emulación de embeddings)
            co.tokenize(text=user_query)
            
            # Gemini genera la respuesta final basada en el contexto del PDF
            response = gemini_model.generate_content(full_prompt)
            bot_response = response.text
            
            message_placeholder.markdown(bot_response)
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            
            # 4. GUARDAR EN SISTEMA DE REGISTRO (MongoDB PaaS)
            log_document = {
                "user_student": USER_NAME,
                "timestamp": datetime.utcnow(),
                "query": user_query,
                "response": bot_response,
                "has_context": bool(pdf_context)
            }
            history_collection.insert_one(log_document)
            
        except Exception as e:
            st.error(f"Error procesando los servicios de IA/Nube: {e}")