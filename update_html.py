import os
import glob
import re

files = glob.glob('*.html')

for file in files:
    if file == 'admin.html': continue

    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Update tailwind.config
    content = content.replace('"primary": "#eccb13"', '"primary": "var(--color-primary, #eccb13)"')
    content = content.replace('"accent-pink": "#f472b6"', '"accent-pink": "var(--color-accent-pink, #f472b6)"')
    content = content.replace('"background-light": "#f8f8f6"', '"background-light": "var(--color-bg-light, #f8f8f6)"')
    content = content.replace('"background-dark": "#221f10"', '"background-dark": "var(--color-bg-dark, #221f10)"')

    # 2. Update Logos
    # looking for <img ... src="favicon.jpg" ...>
    content = re.sub(r'(<img[^>]*src=["\']favicon\.jpg["\'][^>]*)>', r'\1 data-edit="images.logo">', content)
    # the big logo in footer
    content = re.sub(r'(<img[^>]*src=["\']https://lh3.googleusercontent.com/aida-public/[^"\']+["\'][^>]*alt=["\']MC Innovación Financiera Logo["\'])', r'\1 data-edit="images.logo"', content)

    # 3. WhatsApp Links
    content = re.sub(r'(<a[^>]*href=["\']https://wa\.me/message/3BZP6CRYQZLZP1["\'])', r'\1 data-edit-href="https://wa.me/{texts.whatsapp_phone}"', content)

    # 4. WhatsApp Texts in footer
    content = re.sub(r'(<span[^>]*>phone</span>\s*)\+57 320 557 5195', r'\1<span data-edit="texts.whatsapp_phone">+57 320 557 5195</span>', content)

    # 5. Email in footer
    content = re.sub(r'(<span[^>]*>mail</span>\s*)gerencia@mcinnovacionfinanciera\.com', r'\1<span data-edit="texts.contact_email">gerencia@mcinnovacionfinanciera.com</span>', content)

    # 6. Address in footer
    content = re.sub(r'(<span[^>]*>location_on</span>\s*)Carrera 24 #51-21, Bogotá', r'\1<span data-edit="texts.contact_address">Carrera 24 #51-21, Bogotá</span>', content)

    # 8. specific code.html updates
    if file == 'code.html':
        content = re.sub(r'(<h1[^>]*>)\s*Recupera tu estabilidad financiera con <span class="text-primary">asesoría profesional</span>\s*(</h1>)', 
                         r'\1\n                    <span data-edit="texts.hero_title">Recupera tu estabilidad financiera con <span class="text-primary" style="color: var(--color-primary);">asesoría profesional</span></span>\n                \2', 
                         content)
        content = re.sub(r'(<p class="text-lg md:text-xl text-slate-600 dark:text-slate-400 leading-relaxed max-w-xl")>', 
                         r'\1 data-edit="texts.hero_subtitle">', 
                         content)
        content = content.replace('<source src="video_oficina.mp4" type="video/mp4">', '<source src="video_oficina.mp4" type="video/mp4" data-edit="images.hero_video">')

    # 7. Add dynamic.js
    if 'dynamic.js' not in content:
        content = content.replace('</body>', '    <script src="dynamic.js"></script>\n</body>')

    with open(file, 'w', encoding='utf-8') as f:
        f.write(content)

print("Done updating HTML files.")
