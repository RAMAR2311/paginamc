import os, glob, re

files = glob.glob('*.html')
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # regex to replace class attribute of html tag
    if 'bg-slate-950' not in content[:200]: # check near start
        content = re.sub(r'<html([^>]*)class="([^"]*)"', r'<html\1class="\2 bg-slate-950"', content)
        # if html tag has no class, add it
        if '<html lang="es">' in content:
             content = content.replace('<html lang="es">', '<html lang="es" class="bg-slate-950">')
             
    if content != original:
        with open(file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Updated {file}')
