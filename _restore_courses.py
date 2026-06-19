import json

with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Get original curso config
curso = data.get('config', {}).get('texts', {}).get('curso', {})

# Create the first course from existing config data
course1 = {
    "id": "c1",
    "title": curso.get('title', 'Eliminación de Reportes Negativos'),
    "subtitle": curso.get('subtitle', ''),
    "pretitle": curso.get('pretitle', 'Formación Premium'),
    "price": curso.get('price', 150000),
    "wompi_public_key": curso.get('wompi_public_key', ''),
    "cta_text": curso.get('cta_text', 'Compra tu curso ya'),
    "photo": "",
    "benefits": curso.get('benefits', [
        "Fundamentos legales de la Ley 1266 (Habeas Data)",
        "Redacción profesional de derechos de petición",
        "Identificación de reportes eliminables por caducidad",
        "Plantillas listas para usar en Datacrédito y TransUnion",
        "Acciones de Tutela y recursos ante la SIC",
        "Certificado de finalización oficial de MC Innovación"
    ]),
    "modules": [
        {
            "id": "m1",
            "title": "Marco Legal del Habeas Data",
            "lessons": [
                {
                    "id": "l1",
                    "title": "Introducción a la Ley 1266 de 2008",
                    "duration": "15:00",
                    "video_url": "",
                    "description": "Conoce a fondo la Ley 1266 de 2008 y la Ley 2157 de 2021 (Borrón y Cuenta Nueva). Aprende cuáles son tus derechos fundamentales frente a las centrales de riesgo.",
                    "resources": []
                }
            ]
        },
        {
            "id": "m2",
            "title": "Análisis de Reportes",
            "lessons": [
                {
                    "id": "l2",
                    "title": "Cómo interpretar tu historial crediticio",
                    "duration": "20:00",
                    "video_url": "",
                    "description": "Aprende a descargar, interpretar y desglosar tu historial crediticio en Datacrédito y TransUnion. Identifica qué reportes son viables de eliminar por caducidad o suplantación.",
                    "resources": []
                }
            ]
        },
        {
            "id": "m3",
            "title": "Redacción de Derechos de Petición",
            "lessons": [
                {
                    "id": "l3",
                    "title": "Plantillas y estrategias legales",
                    "duration": "25:00",
                    "video_url": "",
                    "description": "Domina la redacción de derechos de petición efectivos dirigidos a bancos, cooperativas y centrales de riesgo. Incluye plantillas descargables y casos reales.",
                    "resources": []
                }
            ]
        },
        {
            "id": "m4",
            "title": "Tutelas y Recursos Legales",
            "lessons": [
                {
                    "id": "l4",
                    "title": "Acciones de Tutela ante la SIC",
                    "duration": "20:00",
                    "video_url": "",
                    "description": "Cuando el derecho de petición no es suficiente, aprende a interponer acciones de tutela y quejas ante la Superintendencia de Industria y Comercio.",
                    "resources": []
                }
            ]
        }
    ]
}

# Create a second course (the user had created "Nuevo Curso")
course2 = {
    "id": "c2",
    "title": "Nuevo Curso",
    "subtitle": "",
    "pretitle": "Formación Premium",
    "price": 0,
    "wompi_public_key": "",
    "cta_text": "Compra tu curso ya",
    "photo": "",
    "benefits": [],
    "modules": []
}

data['courses'] = [course1, course2]

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Restored {len(data['courses'])} courses to data.json!")
