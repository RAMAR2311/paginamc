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
    'config': {
        'colors': {
            'primary': '#eccb13',
            'bgLight': '#f8f8f6',
            'bgDark': '#221f10',
            'accentPink': '#f472b6'
        },
        'texts': {
            'hero_title': 'Recupera tu estabilidad financiera con <span class="text-primary" style="color: var(--color-primary);">asesoría profesional</span>',
            'hero_subtitle': 'En MC Innovación Financiera no solo te guiamos, te proporcionamos las estrategias necesarias para que reactives tu vida y vuelvas a experimentar la tranquilidad financiera.',
            'whatsapp_phone': '573205575195',
            'contact_email': 'gerencia@mcinnovacionfinanciera.com',
            'contact_address': 'Carrera 24 #51-21, Bogotá',
            'home': {
                'services_pretitle': 'Nuestros Servicios Premium',
                'services_title': 'Soluciones integrales a tu medida',
                'service1_title': 'Negociación de deudas',
                'service1_desc': 'Reducimos tus saldos pendientes mediante negociaciones directas con entidades bancarias.',
                'service2_title': 'Reparación de historial',
                'service2_desc': 'Mejoramos tu score en centrales de riesgo para que vuelvas a ser sujeto de crédito.',
                'service3_title': 'Asesoría personalizada',
                'service3_desc': 'Creamos un plan estratégico adaptado a tu realidad financiera y capacidad de pago.',
                'service4_title': 'Educación financiera',
                'service4_desc': 'Te brindamos las herramientas para que nunca vuelvas a caer en situaciones de sobreendeudamiento.',
                'process_pretitle': 'Metodología',
                'process_title': 'Tu camino hacia la libertad financiera en 4 pasos',
                'process1_title': 'Diagnóstico',
                'process1_desc': 'Evaluamos tu situación actual, nivel de deuda y comportamiento crediticio en centrales.',
                'process2_title': 'Análisis',
                'process2_desc': 'Identificamos oportunidades de ahorro y los puntos críticos a negociar con tus acreedores.',
                'process3_title': 'Estrategia',
                'process3_desc': 'Ejecutamos el plan de acción, logrando los mejores descuentos y acuerdos de pago.',
                'process4_title': 'Recuperación',
                'process4_desc': 'Saneamos tu historial y te entregamos un plan de manejo para tu nueva vida financiera.',
                'diff_pretitle': 'Diferenciales',
                'diff_title': '¿Por qué confiar en MC Innovación?',
                'diff_desc': 'Entendemos que detrás de cada deuda hay una persona y una familia buscando paz. No somos una agencia de cobro, somos tu defensa frente al sistema financiero.',
                'diff1_title': 'Transparencia',
                'diff1_desc': 'Sin letras pequeñas ni cobros ocultos.',
                'diff2_title': 'Confidencialidad',
                'diff2_desc': 'Tus datos están protegidos legalmente.',
                'diff3_title': 'Profesionalismo',
                'diff3_desc': 'Consultores certificados por ley.',
                'diff4_title': 'Personalización',
                'diff4_desc': 'Nadie tiene el mismo plan que tú.',
                'diff_item1': 'Acompañamiento legal constante',
                'diff_item2': 'Resultados garantizados por contrato',
                'diff_item3': 'Planes de pago flexibles',
                'testimonials_pretitle': 'Testimonios',
                'testimonials_title': 'Lo que dicen nuestros clientes',
                'cta_title': '¿Listo para volver a dormir tranquilo?',
                'cta_desc': 'Únete a los miles de colombianos que ya recuperaron su libertad financiera con nuestra ayuda.'
            },
            'about': {
                'map_iframe_url': 'https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3976.742407403348!2d-74.07704632469884!3d4.63997629533483!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x8e3f9a3374edcf39%3A0x89ff47d2b9dbc896!2zQ3JhLiAyNCAjNTEtMjEsIFRldXNhcXVpbGxvLCBCb2dvdMOhLCBELkMsIEJvZ290w6E!5e0!3m2!1ses-419!2sco!4v1776468530988!5m2!1ses-419!2sco',
                'mission_vision_title': 'Nuestra <span class="text-primary italic">Misión</span> y <span class="text-primary italic">Visión</span>',
                'mission_vision_desc': 'Nuestra misión es la defensa integral de tus derechos financieros y personales para que recuperes tu libertad crediticia. Nuestra visión es ser tu aliado estratégico en la defensa de derechos, destacando por nuestra empatía, la excelencia de nuestros asesores y un trabajo transparente amparado estrictamente en la Ley 1266 y la Ley 1581.',
                'team_pretitle': 'Nuestro Equipo',
                'team_title': 'Conoce a tus defensores financieros',
                'team_desc': 'Un grupo de 7 expertos comerciales y legales firmemente constituidos como tu aliado estratégico. Trabajamos con transparencia bajo la Ley 1266 de 2008 y la Ley 1581 de 2012 para negociar tus deudas y acompañarte hasta la meta de la libertad crediticia.'
            },
            'footer': {
                'desc': 'Expertos en asesoría financiera y soluciones de deuda en todo el territorio colombiano.',
                'copyright': '© 2024 MC Innovación Financiera. Todos los derechos reservados.',
                'legal': '*Toda nuestra gestión se realiza bajo el estricto marco legal de la Ley 1266 de 2008 (Ley de Habeas Data) y la Ley 1581 de 2012.'
            },
            'aplicativos': {
                'title': 'Nuestros Aplicativos',
                'desc': 'Gestiona y consulta toda la información desde nuestros portales.',
                'empleado_link': 'http://72.62.87.89:8000/auth/login?next=%2Fchat%2F',
                'empleado_hero_title': 'Sistema de Gestión Laboral Integral',
                'empleado_hero_desc': 'Accede a todas las herramientas que necesitas para tu trabajo diario en un solo lugar.',
                'empleado_gateway_title': '¿Eres parte de nuestro equipo?',
                'empleado_gateway_desc': 'Ingresa con tus credenciales corporativas y accede a tu panel personalizado con todas las funcionalidades del sistema.',
                'cliente_link': 'https://www.mcinnovacionfinanciera.cloud/login',
                'cliente_hero_title': '¿Cómo ingresar al portal?',
                'cliente_hero_desc': 'Sigue estos simples pasos para acceder a tu cuenta centralizada.',
                'cliente_step1_title': 'Solicita tus credenciales',
                'cliente_step1_desc': 'Contacta a tu analista financiera para obtener tu usuario y contraseña de acceso al portal.',
                'cliente_step2_title': 'Ingresa al portal',
                'cliente_step2_desc': 'Utiliza las credenciales proporcionadas para acceder por primera vez al sistema.',
                'cliente_step3_title': 'Cambia tu contraseña',
                'cliente_step3_desc': 'Por seguridad, actualiza la contraseña inicial que te fue asignada.',
                'cliente_tools_title': '¿Qué podrás hacer?',
                'cliente_tools_desc': 'Todas las herramientas que necesitas para dar seguimiento a tus casos legales e interactuar con nuestro equipo.',
                'autorizacion_title': 'AUTORIZACIÓN DE TRATAMIENTO DE DATOS PERSONALES',
                'autorizacion_subtitle': 'Y Gestión de Hábeas Data Financiero - MC INNOVACIÓN FINANCIERA',
                'emailjs_public_key': 'zGLHkBswg7hTToZ-Y',
                'emailjs_service_id': 'service_g3cwlis',
                'emailjs_template_autorizacion': 'template_kkv1nxl',
                'pqrs_title': 'Buzón de PQRS',
                'pqrs_subtitle': 'Radique aquí sus Peticiones, Quejas, Reclamos, Sugerencias o Felicitaciones de manera formal y segura. Sus datos personales serán tratados estrictamente conforme a la Ley 1581 de 2012.',
                'emailjs_template_pqrs': 'template_o57vkg3'
            },
            'tienda': {
                'title': 'Tienda MC',
                'desc': 'Adquiere nuestros servicios financieros de forma rápida y segura.'
            }
        },
        'images': {
            'logo': 'favicon.jpg',
            'hero_video': 'video_oficina.mp4'
        }
    },
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
    ],
    'testimonials': [
        { "stars": 5, "text": "Logré negociar una deuda de 30 millones con solo 8 millones de pesos. Mi vida cambió totalmente gracias a MC Innovación.", "name": "Carlos Rodriguez", "location": "Bogotá, D.C.", "photo": "https://lh3.googleusercontent.com/aida-public/AB6AXuA81_t80eYHTFpGWzhGYq7T1ORukxaUmMnlOwldiwqeZ3TZs6RicSvr6sBLJE7LNbsPXvqVOnYFdmRzVkbMRwK8_ELgNC6LI2AW2PElnuwazHAf0epYo67iR8L4qQYWFvrSvBFE2UwTWcpLGeBOYGxE4SvcN0Y_wIFFR2jZllvHLTnScuYEMMtpngLDGxJCJxiFoQXXYRY6w54odANNxya4JZDRh50FE2W3vj2lZXBg0lybyrg5BzmB3qdz913gu80DPUUasCtkRFU" },
        { "stars": 5, "text": "Excelente asesoría. Me explicaron todo de forma clara y sin presiones. Hoy ya tengo mi score de crédito recuperado.", "name": "Ana María Holguín", "location": "Medellín", "photo": "https://lh3.googleusercontent.com/aida-public/AB6AXuBwsp1RjyZ602UOsIqat-ZsgKjJqqDI3Fws01kYywMW4drO4Hy9GtGwNcZx4MXN6CogGCrSsTqQJGxbow5kyjvxIQpTgLvYJ26klQXt4hFMWEWFiDNgnTNQvgaLhgCI5s-MiFyvovNCf-KEJHBHtIBEpfiE_dXBsDnp5VKkYGVH98fy80lvyOBFgHdwaj55t3RqmWfrE0EWKwG6nUSMS4Eqo-JCY6V0fqd7S4Vki2LN3bBvmZScQUYi360I8tAERTYgNfyBIm5AiCs" },
    ],
    'products': [
        { "title": "Diagnóstico Insolvencia Económica", "price": 250000, "category": "asesorias", "desc": "Evaluación técnica para personas naturales no comerciantes que buscan acogerse a la Ley de Insolvencia.", "tag": "Más vendido", "icon": "account_balance" },
        { "title": "Diagnóstico Financiero", "price": 135000, "category": "asesorias", "desc": "Análisis detallado de tu situación financiera actual y capacidad de endeudamiento.", "tag": "", "icon": "analytics" },
        { "title": "Mejora de Score (No Reportados)", "price": 280000, "category": "planes", "desc": "Estrategia de optimización de perfil crediticio para usuarios sin reportes negativos que desean subir su puntaje.", "tag": "Recomendado", "icon": "trending_up" },
        { "title": "Diagnóstico Ayuda del Crédito", "price": 140000, "category": "asesorias", "desc": "Asesoría especializada para la gestión y saneamiento de obligaciones crediticias.", "tag": "", "icon": "contact_support" },
        { "title": "Derecho de Petición (Eliminación de Reportes)", "price": 80000, "category": "tramites", "desc": "Documento legal fundamentado en el Habeas Data (Ley 1266) para solicitar el retiro de reportes negativos injustos o caducados.", "tag": "", "icon": "gavel" },
        { "title": "Curso de Educación Financiera", "price": 310000, "category": "planes", "desc": "Programa de formación integral para el manejo de deudas y alcanzar la libertad crediticia y financiera.", "tag": "", "icon": "school" }
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


@app.route('/api/data/testimonials', methods=['POST'])
def update_testimonials():
    """Actualiza la lista de testimonios y guarda las fotos en el disco"""
    data = load_data()
    testimonials_list = request.json
    
    for item in testimonials_list:
        # Si la foto es nueva (viene en base64), la guardamos en un archivo
        if item.get('photo', '').startswith('data:image'):
            try:
                header, encoded = item.get('photo').split(',', 1)
                ext = header.split('/')[1].split(';')[0]
                filename = f"testimonial_{uuid.uuid4().hex}.{ext}"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(base64.b64decode(encoded))
                
                # Reemplazamos el string base64 super largo por la URL local
                item['photo'] = f'/uploads/{filename}'
            except Exception as e:
                print("Error guardando foto de testimonio:", e)
                
    data['testimonials'] = testimonials_list
    save_data(data)
    return jsonify({"success": True, "message": "Testimonios guardados correctamente"})


@app.route('/api/data/products', methods=['POST'])
def update_products():
    """Actualiza la lista de productos de la tienda y guarda las fotos en el disco"""
    try:
        data = load_data()
        products_list = request.json
        
        for prod in products_list:
            prod['price'] = int(prod.get('price', 0))
            if prod.get('photo', '').startswith('data:image'):
                try:
                    header, encoded = prod['photo'].split(',', 1)
                    ext = header.split('/')[1].split(';')[0]
                    filename = f"product_{uuid.uuid4().hex}.{ext}"
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    
                    with open(filepath, 'wb') as f:
                        f.write(base64.b64decode(encoded))
                    
                    prod['photo'] = f'/uploads/{filename}'
                except Exception as e:
                    print("Error guardando foto de producto:", e)
                    
        data['products'] = products_list
        save_data(data)
        return jsonify({"success": True, "message": "Productos guardados correctamente"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/data/config', methods=['POST'])
def update_config():
    """Actualiza la configuracion general y guarda las fotos en el disco"""
    data = load_data()
    config_data = request.json
    
    # Procesar logo si viene en base64
    logo = config_data.get('images', {}).get('logo', '')
    if logo.startswith('data:image'):
        try:
            header, encoded = logo.split(',', 1)
            ext = header.split('/')[1].split(';')[0]
            filename = f"logo_{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            with open(filepath, 'wb') as f:
                f.write(base64.b64decode(encoded))
            config_data['images']['logo'] = f'/uploads/{filename}'
        except Exception as e:
            print("Error guardando logo:", e)

    # Procesar video de fondo (Hero) si viene en base64
    video = config_data.get('images', {}).get('hero_video', '')
    if video.startswith('data:video'):
        try:
            header, encoded = video.split(',', 1)
            ext = header.split('/')[1].split(';')[0]
            filename = f"hero_video_{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            with open(filepath, 'wb') as f:
                f.write(base64.b64decode(encoded))
            config_data['images']['hero_video'] = f'/uploads/{filename}'
        except Exception as e:
            print("Error guardando video:", e)

    data['config'] = config_data
    save_data(data)
    return jsonify({"success": True, "message": "Configuración guardada correctamente"})

if __name__ == '__main__':
    # Ejecuta el servidor de Flask en el puerto 5000
    app.run(debug=True, port=5000, host='0.0.0.0')
