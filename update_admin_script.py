import re
import os

ADMIN_FILE = 'admin.html'

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

content = read_file(ADMIN_FILE)

# 1. Update HTML UI
# Find where the old text inputs are
html_to_replace = r'''                <!-- Textos -->
                <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6">
                    <h3 class="text-lg font-bold mb-4">Textos y Contacto</h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div class="md:col-span-2">
                            <label class="text-xs font-bold uppercase tracking-widest text-slate-400 mb-1 block">Título Principal (Hero)</label>
                            <input type="text" id="config-text-hero-title" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-primary transition-colors" />
                        </div>
                        <div class="md:col-span-2">
                            <label class="text-xs font-bold uppercase tracking-widest text-slate-400 mb-1 block">Subtítulo (Hero)</label>
                            <textarea id="config-text-hero-subtitle" rows="3" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-primary transition-colors"></textarea>
                        </div>
                        <div>
                            <label class="text-xs font-bold uppercase tracking-widest text-slate-400 mb-1 block">WhatsApp (solo números)</label>
                            <input type="text" id="config-text-wa" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-primary transition-colors" />
                        </div>
                        <div>
                            <label class="text-xs font-bold uppercase tracking-widest text-slate-400 mb-1 block">Correo de Contacto</label>
                            <input type="text" id="config-text-email" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-primary transition-colors" />
                        </div>
                        <div class="md:col-span-2">
                            <label class="text-xs font-bold uppercase tracking-widest text-slate-400 mb-1 block">Dirección</label>
                            <input type="text" id="config-text-address" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-primary transition-colors" />
                        </div>
                    </div>
                </div>'''

new_html = '''                <!-- TEXTOS POR SECCION -->
                <!-- Acordeon Inicio -->
                <details class="bg-slate-900 border border-slate-800 rounded-2xl group overflow-hidden" open>
                    <summary class="p-6 text-lg font-bold cursor-pointer hover:bg-slate-800/50 transition-colors flex justify-between items-center outline-none">
                        Textos: Página Inicio
                        <span class="material-symbols-outlined transform group-open:rotate-180 transition-transform">expand_more</span>
                    </summary>
                    <div class="p-6 border-t border-slate-800 bg-slate-900/50 space-y-6">
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <h4 class="md:col-span-2 text-sm text-primary uppercase font-bold mt-2">Hero & Contacto</h4>
                            <div class="md:col-span-2">
                                <label class="text-xs font-bold uppercase tracking-widest text-slate-400 mb-1 block">Título Principal (Hero)</label>
                                <input type="text" id="config-text-hero-title" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:border-primary transition-colors" />
                            </div>
                            <div class="md:col-span-2">
                                <label class="text-xs font-bold uppercase tracking-widest text-slate-400 mb-1 block">Subtítulo (Hero)</label>
                                <textarea id="config-text-hero-subtitle" rows="3" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:border-primary transition-colors"></textarea>
                            </div>
                            <div>
                                <label class="text-xs font-bold uppercase tracking-widest text-slate-400 mb-1 block">WhatsApp (solo números)</label>
                                <input type="text" id="config-text-wa" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:border-primary transition-colors" />
                            </div>
                            <div>
                                <label class="text-xs font-bold uppercase tracking-widest text-slate-400 mb-1 block">Correo de Contacto</label>
                                <input type="text" id="config-text-email" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:border-primary transition-colors" />
                            </div>
                            <div class="md:col-span-2">
                                <label class="text-xs font-bold uppercase tracking-widest text-slate-400 mb-1 block">Dirección</label>
                                <input type="text" id="config-text-address" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:border-primary transition-colors" />
                            </div>

                            <h4 class="md:col-span-2 text-sm text-primary uppercase font-bold mt-4">Servicios</h4>
                            <div class="md:col-span-1">
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Pre-título</label>
                                <input type="text" id="config-home-services_pretitle" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm" />
                            </div>
                            <div class="md:col-span-1">
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Título</label>
                                <input type="text" id="config-home-services_title" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm" />
                            </div>
                            <div class="md:col-span-1">
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Srv 1: Título</label>
                                <input type="text" id="config-home-service1_title" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm" />
                                <label class="text-xs font-bold text-slate-400 mt-2 mb-1 block">Srv 1: Desc</label>
                                <textarea id="config-home-service1_desc" rows="2" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm"></textarea>
                            </div>
                            <div class="md:col-span-1">
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Srv 2: Título</label>
                                <input type="text" id="config-home-service2_title" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm" />
                                <label class="text-xs font-bold text-slate-400 mt-2 mb-1 block">Srv 2: Desc</label>
                                <textarea id="config-home-service2_desc" rows="2" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm"></textarea>
                            </div>
                            <div class="md:col-span-1">
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Srv 3: Título</label>
                                <input type="text" id="config-home-service3_title" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm" />
                                <label class="text-xs font-bold text-slate-400 mt-2 mb-1 block">Srv 3: Desc</label>
                                <textarea id="config-home-service3_desc" rows="2" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm"></textarea>
                            </div>
                            <div class="md:col-span-1">
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Srv 4: Título</label>
                                <input type="text" id="config-home-service4_title" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm" />
                                <label class="text-xs font-bold text-slate-400 mt-2 mb-1 block">Srv 4: Desc</label>
                                <textarea id="config-home-service4_desc" rows="2" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm"></textarea>
                            </div>

                            <h4 class="md:col-span-2 text-sm text-primary uppercase font-bold mt-4">Proceso</h4>
                            <div class="md:col-span-1">
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Pre-título</label>
                                <input type="text" id="config-home-process_pretitle" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm" />
                            </div>
                            <div class="md:col-span-1">
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Título</label>
                                <input type="text" id="config-home-process_title" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm" />
                            </div>
                            <div class="md:col-span-1">
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Paso 1: Título</label>
                                <input type="text" id="config-home-process1_title" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm" />
                                <label class="text-xs font-bold text-slate-400 mt-2 mb-1 block">Paso 1: Desc</label>
                                <textarea id="config-home-process1_desc" rows="2" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm"></textarea>
                            </div>
                            <div class="md:col-span-1">
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Paso 2: Título</label>
                                <input type="text" id="config-home-process2_title" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm" />
                                <label class="text-xs font-bold text-slate-400 mt-2 mb-1 block">Paso 2: Desc</label>
                                <textarea id="config-home-process2_desc" rows="2" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm"></textarea>
                            </div>
                            <div class="md:col-span-1">
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Paso 3: Título</label>
                                <input type="text" id="config-home-process3_title" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm" />
                                <label class="text-xs font-bold text-slate-400 mt-2 mb-1 block">Paso 3: Desc</label>
                                <textarea id="config-home-process3_desc" rows="2" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm"></textarea>
                            </div>
                            <div class="md:col-span-1">
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Paso 4: Título</label>
                                <input type="text" id="config-home-process4_title" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm" />
                                <label class="text-xs font-bold text-slate-400 mt-2 mb-1 block">Paso 4: Desc</label>
                                <textarea id="config-home-process4_desc" rows="2" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm"></textarea>
                            </div>

                            <h4 class="md:col-span-2 text-sm text-primary uppercase font-bold mt-4">Diferenciales</h4>
                            <div class="md:col-span-1">
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Pre-título</label>
                                <input type="text" id="config-home-diff_pretitle" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm" />
                            </div>
                            <div class="md:col-span-1">
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Título</label>
                                <input type="text" id="config-home-diff_title" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm" />
                            </div>
                            <div class="md:col-span-2">
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Descripción general</label>
                                <textarea id="config-home-diff_desc" rows="2" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm"></textarea>
                            </div>
                            <div class="md:col-span-1">
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Diferencial 1: Título</label>
                                <input type="text" id="config-home-diff1_title" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm" />
                                <label class="text-xs font-bold text-slate-400 mt-2 mb-1 block">Diferencial 1: Desc</label>
                                <textarea id="config-home-diff1_desc" rows="2" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm"></textarea>
                            </div>
                            <div class="md:col-span-1">
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Diferencial 2: Título</label>
                                <input type="text" id="config-home-diff2_title" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm" />
                                <label class="text-xs font-bold text-slate-400 mt-2 mb-1 block">Diferencial 2: Desc</label>
                                <textarea id="config-home-diff2_desc" rows="2" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm"></textarea>
                            </div>
                            <div class="md:col-span-1">
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Diferencial 3: Título</label>
                                <input type="text" id="config-home-diff3_title" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm" />
                                <label class="text-xs font-bold text-slate-400 mt-2 mb-1 block">Diferencial 3: Desc</label>
                                <textarea id="config-home-diff3_desc" rows="2" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm"></textarea>
                            </div>
                            <div class="md:col-span-1">
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Diferencial 4: Título</label>
                                <input type="text" id="config-home-diff4_title" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm" />
                                <label class="text-xs font-bold text-slate-400 mt-2 mb-1 block">Diferencial 4: Desc</label>
                                <textarea id="config-home-diff4_desc" rows="2" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm"></textarea>
                            </div>
                            <div class="md:col-span-2 grid grid-cols-3 gap-2">
                                <div><label class="text-xs font-bold text-slate-400 mb-1 block">Ítem 1</label><input type="text" id="config-home-diff_item1" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-2 py-2 text-xs" /></div>
                                <div><label class="text-xs font-bold text-slate-400 mb-1 block">Ítem 2</label><input type="text" id="config-home-diff_item2" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-2 py-2 text-xs" /></div>
                                <div><label class="text-xs font-bold text-slate-400 mb-1 block">Ítem 3</label><input type="text" id="config-home-diff_item3" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-2 py-2 text-xs" /></div>
                            </div>

                            <h4 class="md:col-span-2 text-sm text-primary uppercase font-bold mt-4">Testimonios & CTA</h4>
                            <div class="md:col-span-1">
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Testimonios Pre-título</label>
                                <input type="text" id="config-home-testimonials_pretitle" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm" />
                            </div>
                            <div class="md:col-span-1">
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Testimonios Título</label>
                                <input type="text" id="config-home-testimonials_title" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm" />
                            </div>
                            <div class="md:col-span-1">
                                <label class="text-xs font-bold text-slate-400 mb-1 block">CTA Final Título</label>
                                <input type="text" id="config-home-cta_title" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm" />
                            </div>
                            <div class="md:col-span-1">
                                <label class="text-xs font-bold text-slate-400 mb-1 block">CTA Final Desc</label>
                                <textarea id="config-home-cta_desc" rows="2" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm"></textarea>
                            </div>
                        </div>
                    </div>
                </details>

                <!-- Acordeon Nosotros -->
                <details class="bg-slate-900 border border-slate-800 rounded-2xl group overflow-hidden">
                    <summary class="p-6 text-lg font-bold cursor-pointer hover:bg-slate-800/50 transition-colors flex justify-between items-center outline-none">
                        Textos: Página Nosotros
                        <span class="material-symbols-outlined transform group-open:rotate-180 transition-transform">expand_more</span>
                    </summary>
                    <div class="p-6 border-t border-slate-800 bg-slate-900/50 space-y-6">
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div class="md:col-span-2">
                                <label class="text-xs font-bold uppercase tracking-widest text-slate-400 mb-1 block">Título Misión y Visión</label>
                                <input type="text" id="config-about-mission_vision_title" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm" />
                            </div>
                            <div class="md:col-span-2">
                                <label class="text-xs font-bold uppercase tracking-widest text-slate-400 mb-1 block">Texto Misión y Visión</label>
                                <textarea id="config-about-mission_vision_desc" rows="4" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm"></textarea>
                            </div>
                            <div class="md:col-span-1">
                                <label class="text-xs font-bold uppercase tracking-widest text-slate-400 mb-1 block">Pre-título Equipo</label>
                                <input type="text" id="config-about-team_pretitle" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm" />
                            </div>
                            <div class="md:col-span-1">
                                <label class="text-xs font-bold uppercase tracking-widest text-slate-400 mb-1 block">Título Equipo</label>
                                <input type="text" id="config-about-team_title" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm" />
                            </div>
                            <div class="md:col-span-2">
                                <label class="text-xs font-bold uppercase tracking-widest text-slate-400 mb-1 block">Descripción Equipo</label>
                                <textarea id="config-about-team_desc" rows="3" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm"></textarea>
                            </div>
                        </div>
                    </div>
                </details>

                <!-- Acordeon Footer -->
                <details class="bg-slate-900 border border-slate-800 rounded-2xl group overflow-hidden">
                    <summary class="p-6 text-lg font-bold cursor-pointer hover:bg-slate-800/50 transition-colors flex justify-between items-center outline-none">
                        Textos: Pie de Página (Footer)
                        <span class="material-symbols-outlined transform group-open:rotate-180 transition-transform">expand_more</span>
                    </summary>
                    <div class="p-6 border-t border-slate-800 bg-slate-900/50 space-y-6">
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div class="md:col-span-2">
                                <label class="text-xs font-bold uppercase tracking-widest text-slate-400 mb-1 block">Descripción Corta</label>
                                <textarea id="config-footer-desc" rows="2" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm"></textarea>
                            </div>
                            <div class="md:col-span-2">
                                <label class="text-xs font-bold uppercase tracking-widest text-slate-400 mb-1 block">Copyright</label>
                                <input type="text" id="config-footer-copyright" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm" />
                            </div>
                            <div class="md:col-span-2">
                                <label class="text-xs font-bold uppercase tracking-widest text-slate-400 mb-1 block">Aviso Legal</label>
                                <textarea id="config-footer-legal" rows="2" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm"></textarea>
                            </div>
                        </div>
                    </div>
                </details>'''

content = content.replace(html_to_replace, new_html)

# 2. Update JS Render
js_render_old = r'''    document.getElementById('config-text-hero-title').value = configData.texts.hero_title;
    document.getElementById('config-text-hero-subtitle').value = configData.texts.hero_subtitle;
    document.getElementById('config-text-wa').value = configData.texts.whatsapp_phone;
    document.getElementById('config-text-email').value = configData.texts.contact_email;
    document.getElementById('config-text-address').value = configData.texts.contact_address;'''

js_render_new = '''    // Base & Contact
    document.getElementById('config-text-hero-title').value = configData.texts.hero_title || '';
    document.getElementById('config-text-hero-subtitle').value = configData.texts.hero_subtitle || '';
    document.getElementById('config-text-wa').value = configData.texts.whatsapp_phone || '';
    document.getElementById('config-text-email').value = configData.texts.contact_email || '';
    document.getElementById('config-text-address').value = configData.texts.contact_address || '';

    // Home
    if (configData.texts.home) {
        ['services_pretitle', 'services_title', 'service1_title', 'service1_desc', 'service2_title', 'service2_desc', 'service3_title', 'service3_desc', 'service4_title', 'service4_desc',
         'process_pretitle', 'process_title', 'process1_title', 'process1_desc', 'process2_title', 'process2_desc', 'process3_title', 'process3_desc', 'process4_title', 'process4_desc',
         'diff_pretitle', 'diff_title', 'diff_desc', 'diff1_title', 'diff1_desc', 'diff2_title', 'diff2_desc', 'diff3_title', 'diff3_desc', 'diff4_title', 'diff4_desc', 'diff_item1', 'diff_item2', 'diff_item3',
         'testimonials_pretitle', 'testimonials_title', 'cta_title', 'cta_desc'
        ].forEach(k => {
            const el = document.getElementById('config-home-' + k);
            if (el) el.value = configData.texts.home[k] || '';
        });
    }
    // About
    if (configData.texts.about) {
        ['mission_vision_title', 'mission_vision_desc', 'team_pretitle', 'team_title', 'team_desc'].forEach(k => {
            const el = document.getElementById('config-about-' + k);
            if (el) el.value = configData.texts.about[k] || '';
        });
    }
    // Footer
    if (configData.texts.footer) {
        ['desc', 'copyright', 'legal'].forEach(k => {
            const el = document.getElementById('config-footer-' + k);
            if (el) el.value = configData.texts.footer[k] || '';
        });
    }'''
content = content.replace(js_render_old, js_render_new)

# 3. Update JS Save
js_save_old = r'''    configData.texts.hero_title = document.getElementById('config-text-hero-title').value;
    configData.texts.hero_subtitle = document.getElementById('config-text-hero-subtitle').value;
    configData.texts.whatsapp_phone = document.getElementById('config-text-wa').value;
    configData.texts.contact_email = document.getElementById('config-text-email').value;
    configData.texts.contact_address = document.getElementById('config-text-address').value;'''

js_save_new = '''    configData.texts.hero_title = document.getElementById('config-text-hero-title').value;
    configData.texts.hero_subtitle = document.getElementById('config-text-hero-subtitle').value;
    configData.texts.whatsapp_phone = document.getElementById('config-text-wa').value;
    configData.texts.contact_email = document.getElementById('config-text-email').value;
    configData.texts.contact_address = document.getElementById('config-text-address').value;

    if (!configData.texts.home) configData.texts.home = {};
    ['services_pretitle', 'services_title', 'service1_title', 'service1_desc', 'service2_title', 'service2_desc', 'service3_title', 'service3_desc', 'service4_title', 'service4_desc',
     'process_pretitle', 'process_title', 'process1_title', 'process1_desc', 'process2_title', 'process2_desc', 'process3_title', 'process3_desc', 'process4_title', 'process4_desc',
     'diff_pretitle', 'diff_title', 'diff_desc', 'diff1_title', 'diff1_desc', 'diff2_title', 'diff2_desc', 'diff3_title', 'diff3_desc', 'diff4_title', 'diff4_desc', 'diff_item1', 'diff_item2', 'diff_item3',
     'testimonials_pretitle', 'testimonials_title', 'cta_title', 'cta_desc'
    ].forEach(k => {
        const el = document.getElementById('config-home-' + k);
        if (el) configData.texts.home[k] = el.value;
    });

    if (!configData.texts.about) configData.texts.about = {};
    ['mission_vision_title', 'mission_vision_desc', 'team_pretitle', 'team_title', 'team_desc'].forEach(k => {
        const el = document.getElementById('config-about-' + k);
        if (el) configData.texts.about[k] = el.value;
    });

    if (!configData.texts.footer) configData.texts.footer = {};
    ['desc', 'copyright', 'legal'].forEach(k => {
        const el = document.getElementById('config-footer-' + k);
        if (el) configData.texts.footer[k] = el.value;
    });'''
content = content.replace(js_save_old, js_save_new)

write_file(ADMIN_FILE, content)
print("Updated admin.html")
