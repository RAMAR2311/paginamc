import os
import json
import uuid
import base64
from flask import Flask, request, jsonify, send_from_directory

# Inicializa la aplicación Flask
# static_folder='.' indica que sirva los archivos estáticos desde el directorio actual
app = Flask(__name__, static_folder='.', static_url_path='')

DATA_FILE = 'data.json'
UPLOAD_FOLDER = 'uploads'

# Crea la carpeta de subidas si no existe
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Datos por defecto si el archivo json no existe
DEFAULT_DATA = {
    'team': [
        { "name": "Alejandro Torres", "role": "Asesor Comercial Senior", "bio": "Especialista en acuerdos comerciales de gran volumen.", "photo": "advisor_portrait_man.png", "whatsapp": "" },
        { "name": "Camila Gómez",    "role": "Asesora Comercial",       "bio": "Manejo de créditos hipotecarios y defensa legal.",   "photo": "advisor_portrait_woman.png", "whatsapp": "" },
        { "name": "Santiago Mejía",  "role": "Asesor Comercial",        "bio": "",  "photo": "advisor_portrait_man.png", "whatsapp": "" },
        { "name": "Diana Ríos",      "role": "Asesora Comercial",       "bio": "",  "photo": "advisor_portrait_woman.png", "whatsapp": "" },
        { "name": "Marlon López",    "role": "Asesor Comercial",        "bio": "",  "photo": "advisor_portrait_man.png", "whatsapp": "" },
        { "name": "Valeria Pineda",  "role": "Asesora Comercial",       "bio": "",  "photo": "advisor_portrait_woman.png", "whatsapp": "" },
        { "name": "Juan Carlos Ruiz","role": "Asesor Comercial",        "bio": "",  "photo": "advisor_portrait_man.png", "whatsapp": "" },
    ],
    'gallery': [
        { "title": "Capacitación Semanal", "subtitle": "Cultura MC",   "photo": "https://images.unsplash.com/photo-1522071823991-b9671e3030e3?auto=format&fit=crop&q=80&w=800" },
        { "title": "Trabajo en Equipo",    "subtitle": "Colaboración", "photo": "https://images.unsplash.com/photo-1517245386807-bb43f82c33c4?auto=format&fit=crop&q=80&w=800" },
        { "title": "Oficinas Centrales",   "subtitle": "Instalaciones","photo": "https://images.unsplash.com/photo-1556761175-b413da4baf72?auto=format&fit=crop&q=80&w=800" },
        { "title": "Atención al Cliente",  "subtitle": "Compromiso",   "photo": "https://images.unsplash.com/photo-1542744173-8e7e53415bb0?auto=format&fit=crop&q=80&w=800" },
    ]
}

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except:
                return DEFAULT_DATA
    return DEFAULT_DATA

def save_data(data):
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

@app.route('/')
def index():
    # Por defecto carga la página principal, asumiendo que sea code.html o index.html
    # Puedes ajustarlo a 'code.html' si esa es tu página de inicio principal.
    return app.send_static_file('code.html') 

@app.route('/<path:path>')
def serve_html(path):
    return app.send_static_file(path)

@app.route('/api/data', methods=['GET'])
def get_data():
    """Retorna los datos actuales para mostrar en la web"""
    return jsonify(load_data())

@app.route('/api/data/team', methods=['POST'])
def update_team():
    """Actualiza la lista del equipo y guarda las fotos en el disco"""
    data = load_data()
    team_list = request.json
    
    for member in team_list:
        # Si la foto es nueva (viene en base64), la guardamos en un archivo
        if member.get('photo', '').startswith('data:image'):
            try:
                header, encoded = member['photo'].split(',', 1)
                ext = header.split('/')[1].split(';')[0]
                filename = f"team_{uuid.uuid4().hex}.{ext}"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(base64.b64decode(encoded))
                
                # Reemplazamos el string base64 super largo por la URL local
                member['photo'] = f'/uploads/{filename}'
            except Exception as e:
                print("Error guardando foto:", e)
                
    data['team'] = team_list
    save_data(data)
    return jsonify({"success": True, "message": "Equipo guardado correctamente"})

@app.route('/api/data/gallery', methods=['POST'])
def update_gallery():
    """Actualiza la galería y guarda las fotos en el disco"""
    data = load_data()
    gallery_list = request.json
    
    for item in gallery_list:
        # Si la foto es nueva (viene en base64), la guardamos en un archivo
        if item.get('photo', '').startswith('data:image'):
            try:
                header, encoded = item.get('photo').split(',', 1)
                ext = header.split('/')[1].split(';')[0]
                filename = f"gallery_{uuid.uuid4().hex}.{ext}"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(base64.b64decode(encoded))
                
                # Reemplazamos el string base64 super largo por la URL local
                item['photo'] = f'/uploads/{filename}'
            except Exception as e:
                print("Error guardando foto:", e)
                
    data['gallery'] = gallery_list
    save_data(data)
    return jsonify({"success": True, "message": "Galería guardada correctamente"})

if __name__ == '__main__':
    # Ejecuta el servidor de Flask en el puerto 5000
    app.run(debug=True, port=5000, host='0.0.0.0')
