import os
import glob

replacements = {
    '[#eccb13]': 'primary',
    '[#221f10]': 'background-dark',
    '[#f8f8f6]': 'background-light',
    '[#f472b6]': 'accent-pink'
}

for filepath in glob.glob('*.html'):
    # Do not replace in admin.html because it uses raw hex for some custom things maybe, or we can replace there too.
    # Actually admin.html has inline styles using `#eccb13`, not tailwind arbitrary values, except maybe in its own config.
    # The regex or text replace should just be literal `[#eccb13]` which is the Tailwind arbitrary value syntax.
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    original_content = content
    for old, new in replacements.items():
        content = content.replace(old, new)
        
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {filepath}")
