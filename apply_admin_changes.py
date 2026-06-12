import os
import re

def update_admin():
    with open('admin.html', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Remove nav-servicios
    nav_servicios_regex = re.compile(r'<button onclick="switchTab\(\'servicios\'\)".*?>.*?</button>', re.DOTALL)
    content = nav_servicios_regex.sub('', content)

    # 2. Extract panel-servicios inner content and remove panel-servicios entirely
    # The panel-servicios is:
    # <!-- ====== SERVICIOS ====== -->
    # <div id="panel-servicios" class="hidden"> ... </div>
    # Actually let's just find the part that needs to be moved.
    
    # Let's extract the "Nuestros Servicios Premium" div:
    servicios_div_regex = re.compile(r'(<div class="bg-slate-900 border border-slate-800 rounded-2xl p-6">\s*<h3 class="text-lg font-bold mb-4">Nuestros Servicios Premium</h3>.*?</div>)', re.DOTALL)
    servicios_match = servicios_div_regex.search(content)
    if servicios_match:
        servicios_html = servicios_match.group(1)
        # Remove it from its original place
        content = content.replace(servicios_html, '')
        
        # Insert it into panel-inicio, say after Textos Hero
        hero_end = content.find('<!-- Proceso -->')
        if hero_end != -1:
            content = content[:hero_end] + '<!-- Servicios Premium -->\n                    ' + servicios_html + '\n\n                    ' + content[hero_end:]
    
    # Remove the remaining panel-servicios
    panel_servicios_regex = re.compile(r'<!-- ====== SERVICIOS ====== -->\s*<div id="panel-servicios".*?</div>\s*</div>\s*</div>', re.DOTALL)
    content = panel_servicios_regex.sub('', content)

    # 3. Move Galería to Nosotros
    # Extract Galería div
    galeria_regex = re.compile(r'(<!-- Galería -->\s*<div class="bg-slate-900 border border-slate-800 rounded-2xl p-6">.*?<div id="gallery-list" class="space-y-4"></div>\s*</div>)', re.DOTALL)
    galeria_match = galeria_regex.search(content)
    if galeria_match:
        galeria_html = galeria_match.group(1)
        content = content.replace(galeria_html, '')
        
        # Insert into panel-nosotros after equipo
        equipo_end = content.find('</div>\n            </div>\n\n            <!-- ====== CONTACTO ====== -->')
        if equipo_end != -1:
            # We want to put it inside the panel-nosotros space-y-8 div
            content = content[:equipo_end] + '\n                    ' + galeria_html + content[equipo_end:]
            
    # 4. Testimonios UI
    # In panel-inicio, replace "Testimonios & CTA" div or just add to it.
    # We will add a dynamic Testimonios list right after the "Testimonios & CTA" settings div.
    testimonios_ui = """
                    <!-- Testimonios Config -->
                    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6">
                        <div class="flex justify-between items-center mb-4">
                            <h3 class="text-lg font-bold">Lista de Testimonios</h3>
                            <button onclick="addTestimonial()" class="flex items-center gap-2 bg-slate-800 text-slate-200 px-4 py-2 rounded-lg font-bold text-sm hover:bg-slate-700 transition-all">
                                <span class="material-symbols-outlined text-sm">add</span> Agregar Testimonio
                            </button>
                        </div>
                        <div id="testimonials-list" class="space-y-4"></div>
                    </div>
    """
    
    # Find "Testimonios y CTA" and insert after it
    cta_end = content.find('<!-- ====== NOSOTROS ====== -->')
    if cta_end != -1:
        # Actually just find the end of the div for CTA
        # Let's insert it before <!-- ====== NOSOTROS ====== -->
        content = content[:cta_end] + testimonios_ui + '\n\n            ' + content[cta_end:]

    # Add JS variables and functions for testimonials
    js_default = "const DEFAULT_TESTIMONIALS = [];\nlet testimonialsData = [];\n"
    content = content.replace('const DEFAULT_TEAM = [', js_default + 'const DEFAULT_TEAM = [')
    
    # Load data
    content = content.replace('galleryData = data.gallery || DEFAULT_GALLERY;', 'galleryData = data.gallery || DEFAULT_GALLERY;\n        testimonialsData = data.testimonials || DEFAULT_TESTIMONIALS;')
    
    # renderAll
    content = content.replace('renderGallery();', 'renderGallery();\n    renderTestimonials();')

    js_logic = """
// ─── TESTIMONIALS RENDERING ────────────────────────────────────
function renderTestimonials() {
    const list = document.getElementById('testimonials-list');
    if(!list) return;
    list.innerHTML = '';
    testimonialsData.forEach((t, i) => {
        const card = document.createElement('div');
        card.className = 'bg-slate-800 border border-slate-700 rounded-xl p-4 flex flex-col md:flex-row gap-4 items-start';
        card.innerHTML = `
            <div class="flex flex-col items-center gap-2 shrink-0">
                <img id="testimonial-photo-preview-${i}" src="${t.photo}" alt="Foto" class="member-photo" onerror="this.style.display='none'" />
                <label for="testimonial-photo-file-${i}" class="upload-label text-[10px] text-slate-400 gap-1 px-2 py-1 w-full text-center">
                    <span class="material-symbols-outlined text-xs">upload</span> Foto
                </label>
                <input type="file" id="testimonial-photo-file-${i}" accept="image/*" onchange="handleTestimonialPhoto(${i}, this)" />
            </div>
            <div class="flex-1 grid grid-cols-1 md:grid-cols-2 gap-3 w-full">
                <div><label class="text-[10px] font-bold uppercase text-slate-400 mb-1 block">Nombre</label><input type="text" id="testimonial-name-${i}" value="${t.name}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs" /></div>
                <div><label class="text-[10px] font-bold uppercase text-slate-400 mb-1 block">Ciudad/Rol</label><input type="text" id="testimonial-city-${i}" value="${t.city}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs" /></div>
                <div class="md:col-span-2"><label class="text-[10px] font-bold uppercase text-slate-400 mb-1 block">Testimonio</label><textarea id="testimonial-text-${i}" rows="2" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs">${t.text}</textarea></div>
            </div>
            <button onclick="removeTestimonial(${i})" class="text-red-400 hover:bg-red-900/30 p-2 rounded-lg shrink-0"><span class="material-symbols-outlined">delete</span></button>
        `;
        list.appendChild(card);
    });
}
function handleTestimonialPhoto(i, input) {
    if (!input.files[0]) return;
    readImageFile(input.files[0], data => {
        testimonialsData[i].photo = data;
        const preview = document.getElementById(`testimonial-photo-preview-${i}`);
        if(preview) { preview.src = data; preview.style.display = 'block'; }
    });
}
function addTestimonial() { testimonialsData.push({ name: 'Nuevo', city: 'Ciudad', text: 'Excelente servicio', photo: '' }); renderTestimonials(); }
function removeTestimonial(i) { if(confirm('¿Eliminar?')){ testimonialsData.splice(i, 1); renderTestimonials(); } }
function collectTestimonials() {
    testimonialsData.forEach((t, i) => {
        t.name = document.getElementById(`testimonial-name-${i}`)?.value || t.name;
        t.city = document.getElementById(`testimonial-city-${i}`)?.value || t.city;
        t.text = document.getElementById(`testimonial-text-${i}`)?.value || t.text;
    });
}
"""
    content = content.replace('// ─── SAVING', js_logic + '\n// ─── SAVING')
    
    # Save promises
    content = content.replace('collectGallery();', 'collectGallery();\n    collectTestimonials();')
    content = content.replace("fetch('/api/data/gallery', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(galleryData) })", "fetch('/api/data/gallery', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(galleryData) }),\n        fetch('/api/data/testimonials', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(testimonialsData) })")
    
    # tabs array
    content = content.replace("const tabs = ['inicio', 'nosotros', 'servicios', 'contacto', 'aplicativos', 'tienda'];", "const tabs = ['inicio', 'nosotros', 'contacto', 'aplicativos', 'tienda'];")

    with open('admin.html', 'w', encoding='utf-8') as f:
        f.write(content)

def update_app_py():
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
        
    endpoint_code = """
@app.route('/api/data/testimonials', methods=['POST'])
def update_testimonials():
    data = load_data()
    t_list = request.json
    
    for item in t_list:
        if item.get('photo', '').startswith('data:image'):
            try:
                header, encoded = item.get('photo').split(',', 1)
                ext = header.split('/')[1].split(';')[0]
                filename = f"testimonial_{uuid.uuid4().hex}.{ext}"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(base64.b64decode(encoded))
                
                item['photo'] = f'/uploads/{filename}'
            except Exception as e:
                print("Error guardando foto:", e)
                
    data['testimonials'] = t_list
    save_data(data)
    return jsonify({"success": True, "message": "Testimonios guardados correctamente"})
"""
    if "def update_testimonials" not in content:
        content = content.replace("@app.route('/api/data/config', methods=['POST'])", endpoint_code + "\n@app.route('/api/data/config', methods=['POST'])")

    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)

def update_code_html():
    with open('code.html', 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the testimonials grid
    grid_regex = re.compile(r'<div class="grid grid-cols-1 md:grid-cols-3 gap-8">.*?</div>\s*</div>\s*</section>', re.DOTALL)
    match = grid_regex.search(content)
    if match:
        new_grid = '<div id="testimonials-container" class="grid grid-cols-1 md:grid-cols-3 gap-8"></div>\n        </div>\n    </section>'
        content = content.replace(match.group(0), new_grid)
        
    with open('code.html', 'w', encoding='utf-8') as f:
        f.write(content)

def update_dynamic_js():
    with open('dynamic.js', 'r', encoding='utf-8') as f:
        content = f.read()

    js_logic = """
            // 4. Testimonials
            if (data.testimonials && data.testimonials.length > 0) {
                const container = document.getElementById('testimonials-container');
                if (container) {
                    container.innerHTML = '';
                    data.testimonials.forEach(t => {
                        const stars = `<div class="flex gap-1 text-primary mb-4">
                            <span class="material-symbols-outlined fill-1">star</span>
                            <span class="material-symbols-outlined fill-1">star</span>
                            <span class="material-symbols-outlined fill-1">star</span>
                            <span class="material-symbols-outlined fill-1">star</span>
                            <span class="material-symbols-outlined fill-1">star</span>
                        </div>`;
                        container.innerHTML += `
                            <div class="p-8 rounded-3xl bg-background-light dark:bg-slate-900/50 border border-slate-200 dark:border-slate-800">
                                ${stars}
                                <p class="text-slate-600 dark:text-slate-400 italic mb-6 leading-relaxed">"${t.text}"</p>
                                <div class="flex items-center gap-4">
                                    <div class="size-12 rounded-full bg-slate-200 overflow-hidden shrink-0">
                                        <img class="w-full h-full object-cover" src="${t.photo}" onerror="this.style.display='none'" />
                                    </div>
                                    <div>
                                        <p class="font-bold text-slate-900 dark:text-white">${t.name}</p>
                                        <p class="text-xs text-slate-500">${t.city}</p>
                                    </div>
                                </div>
                            </div>
                        `;
                    });
                }
            }
"""
    if "data.testimonials" not in content:
        # insert before the catch
        content = content.replace("        })\n        .catch", js_logic + "        })\n        .catch")
        
    with open('dynamic.js', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    update_admin()
    update_app_py()
    update_code_html()
    update_dynamic_js()
    print("All updates complete!")
