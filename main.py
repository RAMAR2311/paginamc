import os
import re
import PyPDF2
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Cargar variables de entorno (.env)
load_dotenv()

# Configurar Modelo Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY no encontrada en el archivo .env")

genai.configure(api_key=GEMINI_API_KEY)

def extract_text_from_knowledge_folder(folder_path="conocimiento"):
    """Recorre la carpeta y extrae texto de txt, html y pdf."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path) # Crear la carpeta si no existe
        return ""
    
    extracted_text = []
    
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        
        if not os.path.isfile(filepath):
            continue
            
        if filename.endswith(".txt"):
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    extracted_text.append(f"--- Documento TXT: {filename} ---\n{f.read()}")
            except Exception as e:
                print(f"Error leyendo TXT {filename}: {e}")
        
        elif filename.endswith(".html"):
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    # Limpieza básica de etiquetas HTML para inyectar a Gemini
                    clean_text = re.sub(r'<[^>]+>', ' ', f.read())
                    extracted_text.append(f"--- Documento HTML: {filename} ---\n{clean_text.strip()}")
            except Exception as e:
                print(f"Error leyendo HTML {filename}: {e}")
                
        elif filename.endswith(".pdf"):
            try:
                pdf_text = ""
                with open(filepath, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            pdf_text += page_text + "\n"
                extracted_text.append(f"--- Documento PDF: {filename} ---\n{pdf_text}")
            except Exception as e:
                print(f"Error leyendo PDF {filename}: {e}")
                
    return "\n\n".join(extracted_text)

# Acumular todo en la variable dinámica requerida
textos_extraidos = extract_text_from_knowledge_folder("conocimiento")



# Instanciar FastAPI
app = FastAPI(title="Sandy - Chatbot RAG MC Innovación")

# Configurar CORS (Para permitir conexión de nuestro widget en HTML local)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estructura de la petición con historial
class ChatMessage(BaseModel):
    role: str # 'user' o 'model'
    parts: str

class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []

# Actualización de instrucciones de Sandy: Personalidad Directa y Breve
SYSTEM_PROMPT = f"""
Identidad: Te llamas Sandy. Eres la asesora experta de 'MC Innovación Financiera'.

Reglas de Estilo Críticas:
0. Primer Contacto: Si es el inicio de la conversación y no sabes el nombre, tu ÚNICA respuesta debe ser un saludo breve y preguntar cómo se llama el usuario. No des más información legal o de servicios hasta que te responda.
1. Brevedad Extrema: Máximo 3 párrafos cortos por respuesta. Usa viñetas (*) para preguntas o listas.
2. No Repetición: No menciones el nombre de la empresa ni leyes en cada mensaje. Cítalas solo si es vital para la explicación.

3. Tono Directo: Trata al usuario de 'tú'. Enfócate en soluciones financieras, no en teoría legal aburrida.
4. Uso del Nombre: Si no conoces el nombre del usuario, pregúntalo en el saludo inicial. Una vez lo sepas, úsalo SOLO al inicio del mensaje para dar cercanía. No lo repitas más.


Lógica de WhatsApp (Solo ofrece hablar con un asesor si ocurre ALGO de esto):
- El usuario usa palabras de urgencia: 'embargo', 'demanda', 'urgente', 'asustado', 'ayuda ya'.
- Detectas que la conversación tiene más de 4 interacciones (revisa el historial).
- El usuario pregunta por un humano o asesor.

Si se cumple la lógica de WhatsApp, pregunta: '¿Te gustaría que te conecte con un asesor por WhatsApp para revisar esto con urgencia?'. 
Si acepta, entrega: https://wa.me/573001234567?text=Hola%20MC%20Innovación,%20vengo%20desde%20el%20chat%20de%20la%20web%20y%20necesito%20asesoría.

BASE DE CONOCIMIENTO DINÁMICA:
{textos_extraidos if textos_extraidos.strip() else '¡Hola! Soy Sandy. Por el momento mi base legal y portafolio están siendo actualizados.'}
"""

# Re-inicializar el modelo con el nuevo prompt
model = genai.GenerativeModel(
    model_name="gemini-flash-latest",
    system_instruction=SYSTEM_PROMPT
)


# Endpoint API POST /chat
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Convertir historial de Pydantic a formato de Gemini
        gemini_history = [
            {"role": m.role, "parts": [m.parts]} for m in request.history
        ]
        
        # Iniciar chat con historial
        chat = model.start_chat(history=gemini_history)
        
        # Enviar mensaje
        response = chat.send_message(request.message)
        
        return {"response": response.text}
    except Exception as e:
        print(f"Error en chat_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

