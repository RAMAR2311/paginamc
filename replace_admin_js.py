import re

with open('admin.html', 'r', encoding='utf-8') as f:
    content = f.read()

new_js = """
let allAdminCourses = [];
let activeAdminCourseIndex = -1;

function renderCoursesList() {
    const list = document.getElementById('cursos-admin-list');
    if (!list) return;
    list.innerHTML = '';
    
    allAdminCourses.forEach((c, idx) => {
        list.innerHTML += `
            <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between group hover:border-primary transition-colors">
                <div class="flex gap-4 items-start mb-4">
                    <img src="${c.photo || 'favicon.jpg'}" class="w-16 h-16 object-cover rounded-xl bg-slate-800 shrink-0" />
                    <div>
                        <h4 class="text-lg font-bold text-white leading-tight mb-1">${c.title || 'Curso Sin Nombre'}</h4>
                        <p class="text-sm text-slate-400">${c.modules ? c.modules.length : 0} módulos</p>
                    </div>
                </div>
                <div class="flex justify-between items-center mt-4 pt-4 border-t border-slate-800">
                    <span class="text-primary font-bold">$${parseInt(c.price || 0).toLocaleString('es-CO')}</span>
                    <div class="flex gap-2">
                        <button onclick="editCourse(${idx})" class="bg-slate-800 text-white px-4 py-2 rounded-lg font-bold text-sm hover:bg-slate-700 transition-all">Editar</button>
                        <button onclick="deleteCourse(${idx})" class="text-red-400 hover:text-red-300 p-2 rounded-lg transition-colors"><span class="material-symbols-outlined text-sm">delete</span></button>
                    </div>
                </div>
            </div>
        `;
    });
}

function createNewCourse() {
    allAdminCourses.push({
        id: 'c' + Date.now(),
        title: 'Nuevo Curso',
        subtitle: '',
        price: 0,
        wompi_public_key: '',
        benefits: [],
        modules: [],
        photo: ''
    });
    renderCoursesList();
    editCourse(allAdminCourses.length - 1);
}

function deleteCourse(idx) {
    if (confirm('¿Seguro que deseas eliminar este curso por completo?')) {
        allAdminCourses.splice(idx, 1);
        renderCoursesList();
    }
}

function editCourse(idx) {
    activeAdminCourseIndex = idx;
    const c = allAdminCourses[idx];
    
    document.getElementById('cursos-list-view').classList.add('hidden');
    document.getElementById('curso-editor-view').classList.remove('hidden');
    
    document.getElementById('config-curso-pretitle').value = c.pretitle || '';
    document.getElementById('config-curso-title').value = c.title || '';
    document.getElementById('config-curso-subtitle').value = c.subtitle || '';
    document.getElementById('config-curso-price').value = c.price || 0;
    document.getElementById('config-curso-wompi-key').value = c.wompi_public_key || '';
    document.getElementById('config-curso-photo-preview').src = c.photo || '';
    
    const photoInput = document.getElementById('config-curso-photo-input');
    photoInput.onchange = async (e) => {
        if (e.target.files && e.target.files[0]) {
            const b64 = await readImageFile(e.target.files[0]);
            allAdminCourses[activeAdminCourseIndex].photo = b64;
            document.getElementById('config-curso-photo-preview').src = b64;
        }
    };
    
    renderCourseBenefits();
    renderCourseModules();
}

function backToCoursesList() {
    saveCurrentCourseConfig();
    document.getElementById('curso-editor-view').classList.add('hidden');
    document.getElementById('cursos-list-view').classList.remove('hidden');
    activeAdminCourseIndex = -1;
    renderCoursesList();
}

function saveCurrentCourseConfig() {
    if (activeAdminCourseIndex === -1) return;
    const c = allAdminCourses[activeAdminCourseIndex];
    c.pretitle = document.getElementById('config-curso-pretitle').value;
    c.title = document.getElementById('config-curso-title').value;
    c.subtitle = document.getElementById('config-curso-subtitle').value;
    c.price = parseInt(document.getElementById('config-curso-price').value) || 0;
    c.wompi_public_key = document.getElementById('config-curso-wompi-key').value;
    c.benefits = collectCourseBenefits();
}

function renderCourseBenefits() {
    const container = document.getElementById('curso-benefits-admin');
    if (!container) return;
    container.innerHTML = '';
    const benefits = allAdminCourses[activeAdminCourseIndex].benefits || [];
    benefits.forEach(b => {
        const row = document.createElement('div');
        row.className = 'flex items-center gap-3 bg-slate-800 p-2 pl-4 rounded-xl';
        row.innerHTML = `
            <span class="material-symbols-outlined text-primary text-lg shrink-0">check_circle</span>
            <input type="text" class="curso-benefit-input flex-1 bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm" value="${b.replace(/"/g, '&quot;')}" />
            <button onclick="this.parentElement.remove()" class="text-red-400 hover:text-red-300 transition-colors shrink-0">
                <span class="material-symbols-outlined text-lg">delete</span>
            </button>
        `;
        container.appendChild(row);
    });
}

function addCourseBenefit() {
    const container = document.getElementById('curso-benefits-admin');
    const row = document.createElement('div');
    row.className = 'flex items-center gap-3 bg-slate-800 p-2 pl-4 rounded-xl';
    row.innerHTML = `
        <span class="material-symbols-outlined text-primary text-lg shrink-0">check_circle</span>
        <input type="text" class="curso-benefit-input flex-1 bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm" value="" placeholder="Nuevo beneficio..." />
        <button onclick="this.parentElement.remove()" class="text-red-400 hover:text-red-300 transition-colors shrink-0">
            <span class="material-symbols-outlined text-lg">delete</span>
        </button>
    `;
    container.appendChild(row);
    row.querySelector('input').focus();
}

function collectCourseBenefits() {
    const inputs = document.querySelectorAll('.curso-benefit-input');
    return Array.from(inputs).map(i => i.value).filter(v => v.trim() !== '');
}

function renderCourseModules() {
    const list = document.getElementById('curso-modules-admin');
    if (!list) return;
    list.innerHTML = '';
    const courseModulesData = allAdminCourses[activeAdminCourseIndex].modules || [];
    
    courseModulesData.forEach((m, mIdx) => {
        const modEl = document.createElement('div');
        modEl.className = 'bg-slate-800 border border-slate-700 rounded-xl p-4';
        
        let lessonsHtml = '';
        if (m.lessons) {
            m.lessons.forEach((l, lIdx) => {
                let resHtml = '';
                if (l.resources && l.resources.length > 0) {
                    l.resources.forEach((r, rIdx) => {
                        resHtml += `
                            <div class="flex items-center gap-2 bg-slate-800 p-1.5 rounded mb-1">
                                <span class="material-symbols-outlined text-[14px] text-primary">description</span>
                                <span class="text-[10px] text-slate-300 truncate flex-1">${r.title}</span>
                                <button onclick="removeCourseLessonResource(${mIdx}, ${lIdx}, ${rIdx})" title="Eliminar recurso" class="text-red-400 hover:text-red-300 p-0.5 rounded transition-colors"><span class="material-symbols-outlined text-[14px]">delete</span></button>
                            </div>
                        `;
                    });
                }
                
                lessonsHtml += `
                    <div class="bg-slate-900 border border-slate-700 rounded-lg p-3 mt-3 flex flex-col gap-2 relative">
                        <button onclick="removeCourseLesson(${mIdx}, ${lIdx})" class="absolute top-2 right-2 text-red-400 hover:bg-red-900/30 p-1 rounded"><span class="material-symbols-outlined text-xs">close</span></button>
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
                            <div><label class="text-[10px] text-slate-400">Título de Lección</label><input type="text" value="${l.title || ''}" onchange="updateCourseLesson(${mIdx}, ${lIdx}, 'title', this.value)" class="w-full bg-slate-800 border border-slate-700 rounded p-1 text-xs" /></div>
                            <div><label class="text-[10px] text-slate-400">Duración</label><input type="text" value="${l.duration || ''}" onchange="updateCourseLesson(${mIdx}, ${lIdx}, 'duration', this.value)" class="w-full bg-slate-800 border border-slate-700 rounded p-1 text-xs" /></div>
                            <div class="md:col-span-2"><label class="text-[10px] text-slate-400">URL del Video</label><input type="text" value="${l.video_url || ''}" onchange="updateCourseLesson(${mIdx}, ${lIdx}, 'video_url', this.value)" class="w-full bg-slate-800 border border-slate-700 rounded p-1 text-xs" /></div>
                            <div class="md:col-span-2"><label class="text-[10px] text-slate-400">Descripción</label><textarea rows="2" onchange="updateCourseLesson(${mIdx}, ${lIdx}, 'description', this.value)" class="w-full bg-slate-800 border border-slate-700 rounded p-1 text-xs">${l.description || ''}</textarea></div>
                            <div class="md:col-span-2 border-t border-slate-700 pt-2 mt-1">
                                <label class="text-[10px] font-bold text-slate-400 block mb-1">Recursos Descargables</label>
                                <div class="mb-2">${resHtml}</div>
                                <div class="flex items-center gap-2">
                                    <input type="file" id="file-res-${mIdx}-${lIdx}" class="hidden" multiple onchange="addCourseLessonResources(${mIdx}, ${lIdx})" />
                                    <button onclick="document.getElementById('file-res-${mIdx}-${lIdx}').click()" class="text-[10px] bg-primary text-slate-900 font-bold px-3 py-1.5 rounded hover:bg-primary/90 transition-colors flex items-center gap-1">
                                        <span class="material-symbols-outlined text-[12px]">upload</span> Subir Archivos
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            });
        }

        modEl.innerHTML = `
            <div class="flex justify-between items-center mb-2">
                <input type="text" value="${m.title || ''}" onchange="updateCourseModule(${mIdx}, 'title', this.value)" class="w-2/3 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm font-bold" placeholder="Título del Módulo" />
                <div class="flex gap-2">
                    <button onclick="addCourseLesson(${mIdx})" class="text-xs bg-slate-700 hover:bg-slate-600 px-3 py-1.5 rounded-lg flex items-center gap-1"><span class="material-symbols-outlined text-[10px]">add</span> Lección</button>
                    <button onclick="removeCourseModule(${mIdx})" class="text-red-400 hover:bg-red-900/30 p-1.5 rounded-lg"><span class="material-symbols-outlined text-sm">delete</span></button>
                </div>
            </div>
            <div class="pl-4 border-l-2 border-slate-700">
                ${lessonsHtml}
            </div>
        `;
        list.appendChild(modEl);
    });
}

function addModule() {
    allAdminCourses[activeAdminCourseIndex].modules.push({ id: 'm' + Date.now(), title: 'Nuevo Módulo', lessons: [] });
    renderCourseModules();
}
function removeCourseModule(mIdx) {
    if (confirm('¿Eliminar módulo?')) { allAdminCourses[activeAdminCourseIndex].modules.splice(mIdx, 1); renderCourseModules(); }
}
function updateCourseModule(mIdx, field, val) { allAdminCourses[activeAdminCourseIndex].modules[mIdx][field] = val; }

function addCourseLesson(mIdx) {
    if (!allAdminCourses[activeAdminCourseIndex].modules[mIdx].lessons) allAdminCourses[activeAdminCourseIndex].modules[mIdx].lessons = [];
    allAdminCourses[activeAdminCourseIndex].modules[mIdx].lessons.push({ id: 'l' + Date.now(), title: 'Nueva Lección', duration: '10:00', video_url: '', description: '', resources: [] });
    renderCourseModules();
}
function removeCourseLesson(mIdx, lIdx) {
    if (confirm('¿Eliminar lección?')) { allAdminCourses[activeAdminCourseIndex].modules[mIdx].lessons.splice(lIdx, 1); renderCourseModules(); }
}
function updateCourseLesson(mIdx, lIdx, field, val) {
    allAdminCourses[activeAdminCourseIndex].modules[mIdx].lessons[lIdx][field] = val;
}

async function addCourseLessonResources(mIdx, lIdx) {
    const input = document.getElementById(`file-res-${mIdx}-${lIdx}`);
    if (!input || !input.files || input.files.length === 0) return;
    
    if (!allAdminCourses[activeAdminCourseIndex].modules[mIdx].lessons[lIdx].resources) {
        allAdminCourses[activeAdminCourseIndex].modules[mIdx].lessons[lIdx].resources = [];
    }
    
    for (let i = 0; i < input.files.length; i++) {
        const file = input.files[i];
        try {
            const base64Url = await readImageFile(file);
            allAdminCourses[activeAdminCourseIndex].modules[mIdx].lessons[lIdx].resources.push({
                title: file.name,
                url: base64Url,
                type: file.name.split('.').pop().toLowerCase()
            });
        } catch(e) { console.error(e); }
    }
    input.value = '';
    renderCourseModules();
}

function removeCourseLessonResource(mIdx, lIdx, rIdx) {
    if (confirm('¿Eliminar recurso?')) {
        allAdminCourses[activeAdminCourseIndex].modules[mIdx].lessons[lIdx].resources.splice(rIdx, 1);
        renderCourseModules();
    }
}
"""

start_marker = '// ─── CURSO BENEFITS ───────────────────────────────────────'
end_marker = '// ─── USERS TABLE ────────────────────────────────────────────'

if start_marker in content and end_marker in content:
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    content = content[:start_idx] + new_js + '\n' + content[end_idx:]
    with open('admin.html', 'w', encoding='utf-8') as f:
        f.write(content)
    print('JS Replaced!')
else:
    print('Markers not found')
