with open('admin.html', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# 1. Update loadData
old_loadData = '''fetch('/api/data?_t=' + Date.now())
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
new_loadData = '''fetch('/api/data?_t=' + Date.now())
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

# 2. Update renderAll
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
    renderCoursesList();
    loadUsers();
}'''
content = content.replace(old_renderAll, new_renderAll)

# 3. Remove old Curso logic from renderConfig
curso_config_pattern = r'// Curso\s*if\s*\(configData\.texts\.curso\).*?renderCourseBenefits\(curso\.benefits\s*\|\|\s*\[\]\);\s*\}'
content = re.sub(curso_config_pattern, '', content, flags=re.DOTALL)

# 4. Remove old Curso Benefits function
curso_benefits_pattern = r'function renderCourseBenefits\(benefits\).*?\}\s*\}'
content = re.sub(curso_benefits_pattern, '', content, flags=re.DOTALL)

# 5. Add new JS logic at the end
with open('replace_admin_js.py', 'r', encoding='utf-8') as js_f:
    js_content = js_f.read()
# Extract the new_js string
js_match = re.search(r'new_js = \"\"\"(.*?)\"\"\"', js_content, flags=re.DOTALL)
if js_match:
    new_js = js_match.group(1)
    # Insert new_js right before closing </script>
    content = content.replace('</script>', new_js + '\n</script>')
    with open('admin.html', 'w', encoding='utf-8') as f:
        f.write(content)
    print('JS Successfully Updated!')
else:
    print('Failed to extract new_js')
