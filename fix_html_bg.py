import os, glob

files = glob.glob('*.html')
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # Update html tag to have bg-slate-950
    if '<html ' in content and 'bg-slate-950' not in content:
        content = content.replace('<html class=\"', '<html class=\"bg-slate-950 ')
        
    if content != original:
        with open(file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Updated {file}')
