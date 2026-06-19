import glob
import os

files = glob.glob('*.html')
replacements = [
    ('href="https://www.facebook.com/share/1H8QWxxbe5/?mibextid=wwXIfr"', 'href="https://www.facebook.com/share/1H8QWxxbe5/?mibextid=wwXIfr" data-edit-href="{texts.social_facebook}"'),
    ('href="https://www.instagram.com/mcinnovacionfinanciera?igsh=dXVubmQ4bGR0aHlu&utm_source=qr"', 'href="https://www.instagram.com/mcinnovacionfinanciera?igsh=dXVubmQ4bGR0aHlu&utm_source=qr" data-edit-href="{texts.social_instagram}"'),
    ('href="https://www.tiktok.com/@mcinnovacionfinanciera?_r=1&_t=ZS-95PGQr4PYXg"', 'href="https://www.tiktok.com/@mcinnovacionfinanciera?_r=1&_t=ZS-95PGQr4PYXg" data-edit-href="{texts.social_tiktok}"')
]

for filepath in files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    modified = False
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            modified = True
            
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Updated {filepath}')
