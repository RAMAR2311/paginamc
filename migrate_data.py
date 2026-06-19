import json
import os

with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

if 'texts' in data.get('config', {}) and 'curso' in data['config']['texts']:
    curso_info = data['config']['texts']['curso']
    
    if 'courses' not in data:
        data['courses'] = []
        
    if not data['courses']:
        course = {
            'id': 'c1',
            'title': curso_info.get('title', 'Eliminación de Reportes'),
            'subtitle': curso_info.get('subtitle', ''),
            'price': curso_info.get('price', 150000),
            'wompi_public_key': curso_info.get('wompi_public_key', ''),
            'benefits': curso_info.get('benefits', []),
            'modules': curso_info.get('modules', []),
            'photo': 'https://images.unsplash.com/photo-1556761175-b413da4baf72?auto=format&fit=crop&q=80&w=800'
        }
        data['courses'].append(course)

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
