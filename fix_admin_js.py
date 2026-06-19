import re

with open('admin.html', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update loadData (HEAD version)
old_loadData = '''let productsData = [];

// Cargar datos desde el backend
fetch('/api/data?_t=' + Date.now())
    .then(res => res.json())
    .then(data => {
        configData = data.config || null;
        teamData = data.team || DEFAULT_TEAM;
        galleryData = data.gallery || DEFAULT_GALLERY;
        testimonialsData = data.testimonials || DEFAULT_TESTIMONIALS;
        productsData = data.products || [];
        if (!document.getElementById('admin-panel').classList.contains('hidden')) {
            renderAll();
        }
    })'''
new_loadData = '''let productsData = [];
let allAdminCourses = [];
let activeAdminCourseIndex = -1;

// Cargar datos desde el backend
fetch('/api/data?_t=' + Date.now())
    .then(res => res.json())
    .then(data => {
        configData = data.config || null;
        teamData = data.team || DEFAULT_TEAM;
        galleryData = data.gallery || DEFAULT_GALLERY;
        testimonialsData = data.testimonials || DEFAULT_TESTIMONIALS;
        productsData = data.products || [];
        allAdminCourses = data.courses || [];
        if (!document.getElementById('admin-panel').classList.contains('hidden')) {
            renderAll();
        }
    })'''
content = content.replace(old_loadData, new_loadData)

# 2. Update renderAll (HEAD version)
old_renderAll = '''function renderAll() {
    renderConfig();
    renderTeam();
    renderGallery();
    renderTestimonials();
    renderProducts();
    loadUsers();
}'''
new_renderAll = '''function renderAll() {
    renderConfig();
    renderTeam();
    renderGallery();
    renderTestimonials();
    renderProducts();
    if (typeof renderCoursesList === 'function') renderCoursesList();
    loadUsers();
}'''
content = content.replace(old_renderAll, new_renderAll)

# 3. Remove old Curso logic from renderConfig
curso_config_pattern = r'// Curso\s*if\s*\(configData\.texts\.curso\).*?renderCourseBenefits\(curso\.benefits\s*\|\|\s*\[\]\);\s*\}'
content = re.sub(curso_config_pattern, '', content, flags=re.DOTALL)

# 4. Remove old Curso Benefits function
curso_benefits_pattern = r'function renderCourseBenefits\(benefits\).*?\}\s*\}'
content = re.sub(curso_benefits_pattern, '', content, flags=re.DOTALL)

# 5. Remove old saveAll for curso
curso_save_pattern = r'// Collect Curso.*?cursoBenefitsData;\s*\}'
content = re.sub(curso_save_pattern, '// Curso\n    if (activeAdminCourseIndex !== -1) saveCurrentCourseConfig();', content, flags=re.DOTALL)

# 6. Update fetch to include courses
old_fetch_products = "fetch('/api/data/products', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(productsData) })"
new_fetch_products = "fetch('/api/data/products', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(productsData) }),\n        fetch('/api/data/courses', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(allAdminCourses) })"
content = content.replace(old_fetch_products, new_fetch_products)

# 7. Append new_js at the end
with open('replace_admin_js.py', 'r', encoding='utf-8') as js_f:
    js_content = js_f.read()
js_match = re.search(r'new_js = \"\"\"(.*?)\"\"\"', js_content, flags=re.DOTALL)
if js_match:
    new_js = js_match.group(1)
    content = content.replace('</script>', new_js + '\n</script>')

with open('admin.html', 'w', encoding='utf-8') as f:
    f.write(content)
print('Successfully fixed!')
