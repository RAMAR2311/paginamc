import re

with open('admin.html', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update fetch(/api/data) load section
old_load = '''courseModulesData = (data.config && data.config.texts && data.config.texts.curso && data.config.texts.curso.modules) ? data.config.texts.curso.modules : [];
        if (!document.getElementById('admin-panel').classList.contains('hidden')) {'''
new_load = '''allAdminCourses = data.courses || [];
        if (!document.getElementById('admin-panel').classList.contains('hidden')) {'''
content = content.replace(old_load, new_load)

# 2. Update renderAll()
old_render = '''    renderProducts();
    loadUsers();
}'''
new_render = '''    renderProducts();
    if(typeof renderCoursesList === 'function') renderCoursesList();
    loadUsers();
}'''
content = content.replace(old_render, new_render)

# 3. Modify saveAll() single course block
curso_save_block = re.search(r'// Curso\n    if \(!configData\.texts\.curso\).*?configData\.texts\.curso\.benefits = collectCourseBenefits\(\);\n', content, re.DOTALL)
if curso_save_block:
    content = content.replace(curso_save_block.group(0), '// Curso\n    if (activeAdminCourseIndex !== -1) saveCurrentCourseConfig();\n')

# 4. Add fetch('/api/data/courses') in saveAll()
old_promise = '''fetch('/api/data/products', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(productsData) })
    ])'''
new_promise = '''fetch('/api/data/products', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(productsData) }),
        fetch('/api/data/courses', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(allAdminCourses) })
    ])'''
content = content.replace(old_promise, new_promise)

# 5. Extract JS from replace_admin_js.py and insert it before loadUsers()
with open('replace_admin_js.py', 'r', encoding='utf-8') as f:
    js_script = f.read()
    new_js_match = re.search(r'new_js = \"\"\"(.*?)\"\"\"', js_script, re.DOTALL)
    if new_js_match:
        new_js = new_js_match.group(1)
        content = content.replace('function loadUsers() {', new_js + '\nfunction loadUsers() {')

with open('admin.html', 'w', encoding='utf-8') as f:
    f.write(content)

print('Admin JS updated perfectly!')
