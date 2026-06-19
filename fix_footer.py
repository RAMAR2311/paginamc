import os, glob

files = glob.glob('*.html')
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # Update body
    if 'h-screen flex flex-col' not in content and 'min-h-screen flex flex-col' not in content and file not in ['admin.html', 'curso-contenido.html']:
        if '<body class=\"' in content:
            content = content.replace('<body class=\"', '<body class=\"min-h-screen flex flex-col ')
    
    # Update footer
    if '<footer ' in content and 'mt-auto' not in content:
        content = content.replace('<footer class=\"', '<footer class=\"mt-auto ')
        
    if content != original:
        with open(file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Updated {file}')
