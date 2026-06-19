import re

with open('admin.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace data loading to include courses
old_load = "let productsData = [];\nlet courseModulesData = [];\n\n// Cargar datos desde el backend\nfetch('/api/data?_t=' + Date.now())\n    .then(res => res.json())\n    .then(data => {\n        configData = data.config || null;\n        teamData = data.team || DEFAULT_TEAM;\n        galleryData = data.gallery || DEFAULT_GALLERY;\n        testimonialsData = data.testimonials || DEFAULT_TESTIMONIALS;\n        productsData = data.products || [];\n        courseModulesData = (data.config && data.config.texts && data.config.texts.curso && data.config.texts.curso.modules) ? data.config.texts.curso.modules : [];\n        if (!document.getElementById('admin-panel').classList.contains('hidden')) {\n            renderAll();\n        }\n    })\n    .catch(err => console.error('Error cargando datos del backend:', err));"

new_load = """let productsData = [];

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
    })
    .catch(err => console.error('Error cargando datos del backend:', err));"""

content = content.replace(old_load, new_load)

# Also update `renderAll()` to include `renderCoursesList()`
old_render = """function renderAll() {
    renderConfig();
    renderTeam();
    renderGallery();
    renderTestimonials();
    renderProducts();
    loadUsers();
}"""

new_render = """function renderAll() {
    renderConfig();
    renderTeam();
    renderGallery();
    renderTestimonials();
    renderProducts();
    renderCoursesList();
    loadUsers();
}"""

content = content.replace(old_render, new_render)

# Now remove the old Curso section from saveAll()
# Replace from `// Curso` to `configData.texts.curso.modules = courseModulesData;`
curso_save_regex = re.compile(r'// Curso\n.*?configData\.texts\.curso\.modules = courseModulesData;', re.DOTALL)
content = curso_save_regex.sub('// Curso\n    if (activeAdminCourseIndex !== -1) saveCurrentCourseConfig();', content)

# Add the fetch for courses in Promise.all
# fetch('/api/data/products', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(productsData) })
old_fetch_products = "fetch('/api/data/products', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(productsData) })"
new_fetch_products = "fetch('/api/data/products', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(productsData) }),\n        fetch('/api/data/courses', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(allAdminCourses) })"

content = content.replace(old_fetch_products, new_fetch_products)

# Also remove `renderCourseBenefits(curso.benefits || []);` and `renderCourseModules();` from `renderConfig`
old_render_curso = """// Curso
    if (configData.texts.curso) {
        const curso = configData.texts.curso;
        document.getElementById('config-curso-pretitle').value = curso.pretitle || '';
        document.getElementById('config-curso-title').value = curso.title || '';
        document.getElementById('config-curso-subtitle').value = curso.subtitle || '';
        document.getElementById('config-curso-price').value = curso.price || 150000;
        document.getElementById('config-curso-wompi-key').value = curso.wompi_public_key || '';
        document.getElementById('config-curso-cta').value = curso.cta_text || '';
        renderCourseBenefits(curso.benefits || []);
        renderCourseModules();
    }"""
content = content.replace(old_render_curso, "")


with open('admin.html', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated successfully!")
