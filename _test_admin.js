
// ─── CONFIG ───────────────────────────────────────────────
const ADMIN_PASSWORD = 'MC2026*'; 

// ─── DEFAULT DATA ─────────────────────────────────────────
const DEFAULT_TEAM = [
    { name: 'Alejandro Torres', role: 'Asesor Comercial Senior', bio: 'Especialista en acuerdos comerciales de gran volumen.', photo: 'advisor_portrait_man.png', whatsapp: '' },
    { name: 'Camila Gómez',    role: 'Asesora Comercial',       bio: 'Manejo de créditos hipotecarios y defensa legal.',   photo: 'advisor_portrait_woman.png', whatsapp: '' },
    { name: 'Santiago Mejía',  role: 'Asesor Comercial',        bio: '',                                                    photo: 'advisor_portrait_man.png', whatsapp: '' }
];
const DEFAULT_GALLERY = [];

// ─── STATE ────────────────────────────────────────────────
const DEFAULT_TESTIMONIALS = [
    { stars: 5, text: "Logré negociar una deuda de 30 millones con solo 8 millones de pesos. Mi vida cambió totalmente gracias a MC Innovación.", name: "Carlos Rodriguez", location: "Bogotá, D.C.", photo: "https://lh3.googleusercontent.com/aida-public/AB6AXu81_t80eYHTFpGWzhGYq7T1ORukxaUmMnlOwldiwqeZ3TZs6RicSvr6sBLJE7LNbsPXvqVOnYFdmRzVkbMRwK8_ELgNC6LI2AW2PElnuwazHAf0epYo67iR8L4qQYWFvrSvBFE2UwTWcpLGeBOYGxE4SvcN0Y_wIFFR2jZllvHLTnScuYEMMtpngLDGxJCJxiFoQXXYRY6w54odANNxya4JZDRh50FE2W3vj2lZXBg0lybyrg5BzmB3qdz913gu80DPUUasCtkRFU" },
    { stars: 5, text: "Excelente asesoría. Me explicaron todo de forma clara y sin presiones. Hoy ya tengo mi score de crédito recuperado.", name: "Ana María Holguín", location: "Medellín", photo: "https://lh3.googleusercontent.com/aida-public/AB6AXuBwsp1RjyZ602UOsIqat-ZsgKjJqqDI3Fws01kYywMW4drO4Hy9GtGwNcZx4MXN6CogGCrSsTqQJGxbow5kyjvxIQpTgLvYJ26klQXt4hFMWEWFiDNgnTNQvgaLhgCI5s-MiFyvovNCf-KEJHBHtIBEpfiE_dXBsDnp5VKkYGVH98fy80lvyOBFgHdwaj55t3RqmWfrE0EWKwG6nUSMS4Eqo-JCY6V0fqd7S4Vki2LN3bBvmZScQUYi360I8tAERTYgNfyBIm5AiCs" },
    { stars: 5, text: "Su profesionalismo y trato humano es lo que más destaco. Te hacen sentir apoyado en todo momento.", name: "Jorge Varela", location: "Cali", photo: "https://lh3.googleusercontent.com/aida-public/AB6AXuDHpXXJobIvptqls2iGvTAzctocBSb5DLMeQEQOMVcBvJJhH8V403kGqDWhbVulaiAybCN61QKzeIawRgo6ZsMww6ql-QjNbCmpXR2J2l_vaWbEdhIGgYTsdNNHJFZdJH20RDoWWODJ8xgIWsot2XYX6Wl9Uyq38Cbkj5ZLHZVDi1rYFYst8DeU4b_HEWXIGfSKfz4_zHMNU6p_O4ntO1L74rZKovvGZoxfxdiJJxZlrxdQvAIzIA5bVI3ut5tkD9f18LyGmE9q_fg" }
];

let configData = null;
let teamData = DEFAULT_TEAM;
let galleryData = DEFAULT_GALLERY;
let testimonialsData = DEFAULT_TESTIMONIALS;
let productsData = [];

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
    .catch(err => console.error('Error cargando datos del backend:', err));

// ─── AUTH ─────────────────────────────────────────────────
function doLogin() {
    const pw = document.getElementById('password-input').value;
    if (pw === ADMIN_PASSWORD) {
        document.getElementById('login-screen').classList.add('hidden');
        document.getElementById('admin-panel').classList.remove('hidden');
        if(configData) renderAll();
    } else {
        document.getElementById('login-error').classList.remove('hidden');
        document.getElementById('password-input').value = '';
    }
}

function doLogout() {
    document.getElementById('admin-panel').classList.add('hidden');
    document.getElementById('login-screen').classList.remove('hidden');
    document.getElementById('password-input').value = '';
}

// ─── TABS ─────────────────────────────────────────────────
const tabs = ['inicio', 'nosotros', 'servicios', 'contacto', 'aplicativos', 'tienda', 'curso', 'usuarios'];
function switchTab(tab) {
    tabs.forEach(t => {
        document.getElementById('panel-' + t).classList.add('hidden');
        const nav = document.getElementById('nav-' + t);
        nav.className = 'nav-inactive w-full flex items-center gap-3 px-4 py-3 rounded-xl font-bold text-sm transition-colors';
    });
    
    document.getElementById('panel-' + tab).classList.remove('hidden');
    document.getElementById('nav-' + tab).className = 'nav-active w-full flex items-center gap-3 px-4 py-3 rounded-xl font-bold text-sm transition-colors';
}

// ─── TOAST ────────────────────────────────────────────────
function showToast(msg = 'Guardado con éxito') {
    const t = document.getElementById('toast');
    document.getElementById('toast-msg').textContent = msg;
    t.classList.remove('hidden');
    t.classList.add('flex');
    setTimeout(() => { t.classList.add('hidden'); t.classList.remove('flex'); }, 3000);
}

// ─── IMAGE HELPER (CON COMPRESIÓN) ────────────────────────
function readImageFile(file, callback) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const img = new Image();
        img.onload = function() {
            const MAX_WIDTH = 1200;
            const MAX_HEIGHT = 1200;
            let width = img.width;
            let height = img.height;
            if (width > height) {
                if (width > MAX_WIDTH) { height *= MAX_WIDTH / width; width = MAX_WIDTH; }
            } else {
                if (height > MAX_HEIGHT) { width *= MAX_HEIGHT / height; height = MAX_HEIGHT; }
            }
            const canvas = document.createElement('canvas');
            canvas.width = width; canvas.height = height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, width, height);
            const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
            callback(dataUrl);
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// ─── RENDERING ─────────────────────────────────────
function renderAll() {
    renderConfig();
    renderTeam();
    renderGallery();
    renderTestimonials();
    renderProducts();
    if(typeof renderCoursesList === 'function') renderCoursesList();
    loadUsers();
}

function renderConfig() {
    if(!configData) return;
    document.getElementById('config-color-primary').value = configData.colors.primary;
    document.getElementById('config-color-accent').value = configData.colors.accentPink;
    document.getElementById('config-color-bgLight').value = configData.colors.bgLight;
    document.getElementById('config-color-bgDark').value = configData.colors.bgDark;
    
    document.getElementById('config-text-hero-title').value = configData.texts.hero_title || '';
    document.getElementById('config-text-hero-subtitle').value = configData.texts.hero_subtitle || '';
    document.getElementById('config-text-wa').value = configData.texts.whatsapp_phone || '';
    document.getElementById('config-text-email').value = configData.texts.contact_email || '';
    document.getElementById('config-text-address').value = configData.texts.contact_address || '';

    if (configData.texts.home) {
        ['services_pretitle', 'services_title', 'service1_title', 'service1_desc', 'service2_title', 'service2_desc', 'service3_title', 'service3_desc', 'service4_title', 'service4_desc',
         'process_pretitle', 'process_title', 'process1_title', 'process1_desc', 'process2_title', 'process2_desc', 'process3_title', 'process3_desc', 'process4_title', 'process4_desc',
         'diff_pretitle', 'diff_title', 'diff_desc', 'diff1_title', 'diff1_desc', 'diff2_title', 'diff2_desc', 'diff3_title', 'diff3_desc', 'diff4_title', 'diff4_desc', 'diff_item1', 'diff_item2', 'diff_item3',
         'testimonials_pretitle', 'testimonials_title', 'cta_title', 'cta_desc'].forEach(k => {
            const el = document.getElementById('config-home-' + k);
            if (el) el.value = configData.texts.home[k] || '';
        });
    }
    if (configData.texts.about) {
        ['mission_vision_title', 'mission_vision_desc', 'team_pretitle', 'team_title', 'team_desc', 'map_iframe_url'].forEach(k => {
            const el = document.getElementById('config-about-' + k);
            if (el) el.value = configData.texts.about[k] || '';
        });
    }
    if (configData.texts.footer) {
        ['desc', 'copyright', 'legal'].forEach(k => {
            const el = document.getElementById('config-footer-' + k);
            if (el) el.value = configData.texts.footer[k] || '';
        });
    }
    if (configData.texts.aplicativos) {
        document.getElementById('config-aplicativos-title').value = configData.texts.aplicativos.title || '';
        document.getElementById('config-aplicativos-desc').value = configData.texts.aplicativos.desc || '';
        
        const apps = configData.texts.aplicativos;
        const fields = [
            'empleado_link', 'empleado_hero_title', 'empleado_hero_desc', 'empleado_gateway_title', 'empleado_gateway_desc',
            'cliente_link', 'cliente_hero_title', 'cliente_hero_desc',
            'cliente_step1_title', 'cliente_step1_desc', 'cliente_step2_title', 'cliente_step2_desc', 'cliente_step3_title', 'cliente_step3_desc',
            'cliente_tools_title', 'cliente_tools_desc',
            'autorizacion_title', 'autorizacion_subtitle',
            'emailjs_public_key', 'emailjs_service_id', 'emailjs_template_autorizacion',
            'pqrs_title', 'pqrs_subtitle', 'emailjs_template_pqrs'
        ];
        fields.forEach(f => {
            const el = document.getElementById('config-aplicativos-' + f.replace(/_/g, '-'));
            if (el) el.value = apps[f] || '';
        });
    }
    if (configData.texts.tienda) {
        document.getElementById('config-tienda-title').value = configData.texts.tienda.title || '';
        document.getElementById('config-tienda-desc').value = configData.texts.tienda.desc || '';
    }
    // Curso
    if (configData.texts.curso) {
        const curso = configData.texts.curso;
        document.getElementById('config-curso-pretitle').value = curso.pretitle || '';
        document.getElementById('config-curso-title').value = curso.title || '';
        document.getElementById('config-curso-subtitle').value = curso.subtitle || '';
        document.getElementById('config-curso-price').value = curso.price || 150000;
        document.getElementById('config-curso-wompi-key').value = curso.wompi_public_key || '';
        document.getElementById('config-curso-cta').value = curso.cta_text || '';
        renderCourseBenefits(curso.benefits || []);
    }
    
    document.getElementById('config-logo-preview').src = configData.images.logo;
    document.getElementById('config-video-preview').src = configData.images.hero_video || '';
}

function handleLogoPhoto(input) {
    if (!input.files[0]) return;
    const reader = new FileReader();
    reader.onload = function(e) {
        if(!configData) return;
        configData.images.logo = e.target.result;
        document.getElementById('config-logo-preview').src = e.target.result;
    };
    reader.readAsDataURL(input.files[0]);
}

function handleVideoFile(input) {
    if (!input.files[0]) return;
    if (input.files[0].size > 20 * 1024 * 1024) {
        alert("El video es muy pesado (más de 20MB). Podría tardar bastante en guardarse.");
    }
    const reader = new FileReader();
    reader.onload = function(e) {
        if(!configData) return;
        configData.images.hero_video = e.target.result;
        document.getElementById('config-video-preview').src = e.target.result;
    };
    reader.readAsDataURL(input.files[0]);
}

// ─── TEAM RENDERING ───────────────────────────────────────
function renderTeam() {
    const list = document.getElementById('team-list');
    list.innerHTML = '';
    teamData.forEach((m, i) => {
        const card = document.createElement('div');
        card.className = 'bg-slate-800 border border-slate-700 rounded-xl p-4 flex flex-col md:flex-row gap-4 items-start';
        card.innerHTML = `
            <div class="flex flex-col items-center gap-2 shrink-0">
                <img id="team-photo-preview-${i}" src="${m.photo}" alt="Foto" class="member-photo" onerror="this.src='advisor_portrait_man.png'"/>
                <label for="team-photo-file-${i}" class="upload-label text-[10px] text-slate-400 gap-1 px-2 py-1 w-full text-center">
                    <span class="material-symbols-outlined text-xs">upload</span> Foto
                </label>
                <input type="file" id="team-photo-file-${i}" accept="image/*" onchange="handleTeamPhoto(${i}, this)" />
            </div>
            <div class="flex-1 grid grid-cols-1 md:grid-cols-2 gap-3 w-full">
                <div><label class="text-[10px] font-bold uppercase text-slate-400 mb-1 block">Nombre</label><input type="text" id="team-name-${i}" value="${m.name}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs" /></div>
                <div><label class="text-[10px] font-bold uppercase text-slate-400 mb-1 block">Cargo</label><input type="text" id="team-role-${i}" value="${m.role}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs" /></div>
                <div><label class="text-[10px] font-bold uppercase text-slate-400 mb-1 block">Bio</label><input type="text" id="team-bio-${i}" value="${m.bio}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs" /></div>
                <div><label class="text-[10px] font-bold uppercase text-slate-400 mb-1 block">WhatsApp</label><input type="text" id="team-wa-${i}" value="${m.whatsapp || ''}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs" /></div>
            </div>
            <button onclick="removeMember(${i})" class="text-red-400 hover:bg-red-900/30 p-2 rounded-lg shrink-0"><span class="material-symbols-outlined">delete</span></button>
        `;
        list.appendChild(card);
    });
}
function handleTeamPhoto(i, input) {
    if (!input.files[0]) return;
    readImageFile(input.files[0], data => {
        teamData[i].photo = data;
        document.getElementById(`team-photo-preview-${i}`).src = data;
    });
}
function addMember() { teamData.push({ name: 'Nuevo', role: 'Rol', bio: '', photo: 'advisor_portrait_man.png', whatsapp: '' }); renderTeam(); }
function removeMember(i) { if(confirm('¿Eliminar?')){ teamData.splice(i, 1); renderTeam(); } }
function collectTeam() {
    teamData.forEach((m, i) => {
        m.name = document.getElementById(`team-name-${i}`)?.value || m.name;
        m.role = document.getElementById(`team-role-${i}`)?.value || m.role;
        m.bio = document.getElementById(`team-bio-${i}`)?.value || '';
        m.whatsapp = document.getElementById(`team-wa-${i}`)?.value || '';
    });
}

// ─── GALLERY RENDERING ────────────────────────────────────
function renderGallery() {
    const list = document.getElementById('gallery-list');
    list.innerHTML = '';
    galleryData.forEach((g, i) => {
        const card = document.createElement('div');
        card.className = 'bg-slate-800 border border-slate-700 rounded-xl p-4 flex flex-col md:flex-row gap-4 items-start';
        card.innerHTML = `
            <div class="flex flex-col items-center gap-2 shrink-0">
                <img id="gallery-photo-preview-${i}" src="${g.photo}" alt="Foto" class="gallery-thumb" onerror="this.style.display='none'" />
                <label for="gallery-photo-file-${i}" class="upload-label text-[10px] text-slate-400 gap-1 px-2 py-1 w-full text-center">
                    <span class="material-symbols-outlined text-xs">upload</span> Foto
                </label>
                <input type="file" id="gallery-photo-file-${i}" accept="image/*" onchange="handleGalleryPhoto(${i}, this)" />
            </div>
            <div class="flex-1 grid grid-cols-1 md:grid-cols-2 gap-3 w-full">
                <div><label class="text-[10px] font-bold uppercase text-slate-400 mb-1 block">Título</label><input type="text" id="gallery-title-${i}" value="${g.title}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs" /></div>
                <div><label class="text-[10px] font-bold uppercase text-slate-400 mb-1 block">Subtítulo</label><input type="text" id="gallery-subtitle-${i}" value="${g.subtitle}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs" /></div>
            </div>
            <button onclick="removeGalleryItem(${i})" class="text-red-400 hover:bg-red-900/30 p-2 rounded-lg shrink-0"><span class="material-symbols-outlined">delete</span></button>
        `;
        list.appendChild(card);
    });
}
function handleGalleryPhoto(i, input) {
    if (!input.files[0]) return;
    readImageFile(input.files[0], data => {
        galleryData[i].photo = data;
        const preview = document.getElementById(`gallery-photo-preview-${i}`);
        preview.src = data; preview.style.display = 'block';
    });
}
function addGalleryItem() { galleryData.push({ title: 'Momento', subtitle: 'Desc', photo: '' }); renderGallery(); }
function removeGalleryItem(i) { if(confirm('¿Eliminar?')){ galleryData.splice(i, 1); renderGallery(); } }
function collectGallery() {
    galleryData.forEach((g, i) => {
        g.title = document.getElementById(`gallery-title-${i}`)?.value || g.title;
        g.subtitle = document.getElementById(`gallery-subtitle-${i}`)?.value || g.subtitle;
    });
}

// ─── TESTIMONIALS RENDERING ────────────────────────────────────
function renderTestimonials() {
    const list = document.getElementById('testimonials-list');
    if (!list) return;
    list.innerHTML = '';
    testimonialsData.forEach((t, i) => {
        const card = document.createElement('div');
        card.className = 'bg-slate-800 border border-slate-700 rounded-xl p-4 flex flex-col md:flex-row gap-4 items-start';
        card.innerHTML = `
            <div class="flex flex-col items-center gap-2 shrink-0">
                <img id="testimonial-photo-preview-${i}" src="${t.photo || 'https://lh3.googleusercontent.com/aida-public/AB6AXu81_t80eYHTFpGWzhGYq7T1ORukxaUmMnlOwldiwqeZ3TZs6RicSvr6sBLJE7LNbsPXvqVOnYFdmRzVkbMRwK8_ELgNC6LI2AW2PElnuwazHAf0epYo67iR8L4qQYWFvrSvBFE2UwTWcpLGeBOYGxE4SvcN0Y_wIFFR2jZllvHLTnScuYEMMtpngLDGxJCJxiFoQXXYRY6w54odANNxya4JZDRh50FE2W3vj2lZXBg0lybyrg5BzmB3qdz913gu80DPUUasCtkRFU'}" alt="Foto" class="gallery-thumb" onerror="this.src='https://lh3.googleusercontent.com/aida-public/AB6AXu81_t80eYHTFpGWzhGYq7T1ORukxaUmMnlOwldiwqeZ3TZs6RicSvr6sBLJE7LNbsPXvqVOnYFdmRzVkbMRwK8_ELgNC6LI2AW2PElnuwazHAf0epYo67iR8L4qQYWFvrSvBFE2UwTWcpLGeBOYGxE4SvcN0Y_wIFFR2jZllvHLTnScuYEMMtpngLDGxJCJxiFoQXXYRY6w54odANNxya4JZDRh50FE2W3vj2lZXBg0lybyrg5BzmB3qdz913gu80DPUUasCtkRFU'" />
                <label for="testimonial-photo-file-${i}" class="upload-label text-[10px] text-slate-400 gap-1 px-2 py-1 w-full text-center">
                    <span class="material-symbols-outlined text-xs">upload</span> Foto
                </label>
                <input type="file" id="testimonial-photo-file-${i}" accept="image/*" onchange="handleTestimonialPhoto(${i}, this)" />
            </div>
            <div class="flex-1 grid grid-cols-1 md:grid-cols-3 gap-3 w-full">
                <div class="md:col-span-1">
                    <label class="text-[10px] font-bold uppercase text-slate-400 mb-1 block">Nombre</label>
                    <input type="text" id="testimonial-name-${i}" value="${t.name}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs" />
                </div>
                <div class="md:col-span-1">
                    <label class="text-[10px] font-bold uppercase text-slate-400 mb-1 block">Ubicación</label>
                    <input type="text" id="testimonial-location-${i}" value="${t.location}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs" />
                </div>
                <div class="md:col-span-1">
                    <label class="text-[10px] font-bold uppercase text-slate-400 mb-1 block">Estrellas (1-5)</label>
                    <input type="number" id="testimonial-stars-${i}" min="1" max="5" value="${t.stars || 5}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs" />
                </div>
                <div class="md:col-span-3">
                    <label class="text-[10px] font-bold uppercase text-slate-400 mb-1 block">Texto del Testimonio</label>
                    <textarea id="testimonial-text-${i}" rows="2" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs">${t.text}</textarea>
                </div>
            </div>
            <button onclick="removeTestimonial(${i})" class="text-red-400 hover:bg-red-900/30 p-2 rounded-lg shrink-0"><span class="material-symbols-outlined">delete</span></button>
        `;
        list.appendChild(card);
    });
}
function handleTestimonialPhoto(i, input) {
    if (!input.files[0]) return;
    readImageFile(input.files[0], data => {
        testimonialsData[i].photo = data;
        const preview = document.getElementById(`testimonial-photo-preview-${i}`);
        preview.src = data; preview.style.display = 'block';
    });
}
function addTestimonial() { testimonialsData.push({ stars: 5, text: 'Excelente servicio.', name: 'Cliente Satisfecho', location: 'Colombia', photo: '' }); renderTestimonials(); }
function removeTestimonial(i) { if(confirm('¿Eliminar testimonio?')){ testimonialsData.splice(i, 1); renderTestimonials(); } }
function collectTestimonials() {
    testimonialsData.forEach((t, i) => {
        t.name = document.getElementById(`testimonial-name-${i}`)?.value || t.name;
        t.location = document.getElementById(`testimonial-location-${i}`)?.value || t.location;
        t.stars = parseInt(document.getElementById(`testimonial-stars-${i}`)?.value) || 5;
        t.text = document.getElementById(`testimonial-text-${i}`)?.value || t.text;
    });
}

// ─── PRODUCTS RENDERING ────────────────────────────────────
function renderProducts() {
    const list = document.getElementById('products-list');
    if (!list) return;
    list.innerHTML = '';
    
    // Icon options
    const icons = ['account_balance', 'analytics', 'trending_up', 'contact_support', 'gavel', 'school', 'payments', 'description', 'shield', 'assignment_ind', 'credit_score', 'account_balance_wallet'];
    
    productsData.forEach((p, i) => {
        const card = document.createElement('div');
        card.className = 'bg-slate-800 border border-slate-700 rounded-xl p-4 flex flex-col md:flex-row gap-4 items-start';
        
        let iconOptionsHtml = icons.map(ico => `<option value="${ico}" ${p.icon === ico ? 'selected' : ''}>${ico}</option>`).join('');
        
        const photoUrl = p.photo || '';
        
        card.innerHTML = `
            <div class="flex flex-col items-center gap-2 shrink-0">
                <img id="product-photo-preview-${i}" src="${photoUrl}" alt="Foto" class="gallery-thumb" style="display: ${photoUrl ? 'block' : 'none'}; width: 120px; height: 80px; object-fit: cover;" />
                <div id="product-photo-placeholder-${i}" class="gallery-thumb border-2 border-dashed border-slate-700 flex flex-col items-center justify-center text-[10px] text-slate-500" style="display: ${photoUrl ? 'none' : 'flex'}; width: 120px; height: 80px; border-radius: .75rem;">
                    <span class="material-symbols-outlined text-lg mb-1">image</span>
                    Subir Imagen
                </div>
                <label for="product-photo-file-${i}" class="upload-label text-[10px] text-slate-400 gap-1 px-2 py-1 w-full text-center">
                    <span class="material-symbols-outlined text-xs">upload</span> Foto
                </label>
                <input type="file" id="product-photo-file-${i}" accept="image/*" onchange="handleProductPhoto(${i}, this)" />
                ${photoUrl ? `<button onclick="removeProductPhoto(${i})" class="text-xs text-red-400 hover:underline">Quitar foto</button>` : ''}
            </div>
            <div class="flex-1 grid grid-cols-1 md:grid-cols-3 gap-3 w-full">
                <div class="md:col-span-1">
                    <label class="text-[10px] font-bold uppercase text-slate-400 mb-1 block">Título</label>
                    <input type="text" id="product-title-${i}" value="${p.title}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs" />
                </div>
                <div class="md:col-span-1">
                    <label class="text-[10px] font-bold uppercase text-slate-400 mb-1 block">Precio (COP)</label>
                    <input type="number" id="product-price-${i}" value="${p.price}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs" />
                </div>
                <div class="md:col-span-1">
                    <label class="text-[10px] font-bold uppercase text-slate-400 mb-1 block">Categoría</label>
                    <select id="product-category-${i}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white">
                        <option value="asesorias" ${p.category === 'asesorias' ? 'selected' : ''}>Asesorías</option>
                        <option value="tramites" ${p.category === 'tramites' ? 'selected' : ''}>Trámites Legales</option>
                        <option value="planes" ${p.category === 'planes' ? 'selected' : ''}>Planes Integrales</option>
                    </select>
                </div>
                <div class="md:col-span-1">
                    <label class="text-[10px] font-bold uppercase text-slate-400 mb-1 block">Etiqueta (Opcional)</label>
                    <select id="product-tag-${i}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white">
                        <option value="" ${p.tag === '' ? 'selected' : ''}>Ninguna</option>
                        <option value="Más vendido" ${p.tag === 'Más vendido' ? 'selected' : ''}>Más vendido</option>
                        <option value="Recomendado" ${p.tag === 'Recomendado' ? 'selected' : ''}>Recomendado</option>
                    </select>
                </div>
                <div class="md:col-span-1">
                    <label class="text-[10px] font-bold uppercase text-slate-400 mb-1 block">Icono de Fallback (Material Symbols)</label>
                    <select id="product-icon-${i}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white">
                        ${iconOptionsHtml}
                    </select>
                </div>
                <div class="md:col-span-3">
                    <label class="text-[10px] font-bold uppercase text-slate-400 mb-1 block">Descripción del Servicio</label>
                    <textarea id="product-desc-${i}" rows="2" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs">${p.desc}</textarea>
                </div>
            </div>
            <button onclick="removeProduct(${i})" class="text-red-400 hover:bg-red-900/30 p-2 rounded-lg shrink-0 mt-4 md:mt-0"><span class="material-symbols-outlined">delete</span></button>
        `;
        list.appendChild(card);
    });
}
function handleProductPhoto(i, input) {
    if (!input.files[0]) return;
    readImageFile(input.files[0], data => {
        productsData[i].photo = data;
        renderProducts();
    });
}
function removeProductPhoto(i) {
    productsData[i].photo = '';
    renderProducts();
}
function addProduct() { 
    productsData.push({ title: 'Nuevo Servicio', price: 100000, category: 'asesorias', desc: 'Descripción del servicio.', tag: '', icon: 'analytics', photo: '' }); 
    renderProducts(); 
}
function removeProduct(i) { 
    if(confirm('¿Eliminar este servicio?')){ 
        productsData.splice(i, 1); 
        renderProducts(); 
    } 
}
function collectProducts() {
    productsData.forEach((p, i) => {
        p.title = document.getElementById(`product-title-${i}`)?.value || p.title;
        p.price = parseInt(document.getElementById(`product-price-${i}`)?.value) || 0;
        p.category = document.getElementById(`product-category-${i}`)?.value || p.category;
        p.tag = document.getElementById(`product-tag-${i}`)?.value || '';
        p.icon = document.getElementById(`product-icon-${i}`)?.value || p.icon;
        p.desc = document.getElementById(`product-desc-${i}`)?.value || p.desc;
    });
}

// ─── SAVING ──────────────────────────────────────────────
function saveAll() {
    if(!configData) return;
    showToast('Guardando todo, por favor espera...');

    // Collect Config
    configData.colors.primary = document.getElementById('config-color-primary').value;
    configData.colors.accentPink = document.getElementById('config-color-accent').value;
    configData.colors.bgLight = document.getElementById('config-color-bgLight').value;
    configData.colors.bgDark = document.getElementById('config-color-bgDark').value;
    
    configData.texts.hero_title = document.getElementById('config-text-hero-title').value;
    configData.texts.hero_subtitle = document.getElementById('config-text-hero-subtitle').value;
    configData.texts.whatsapp_phone = document.getElementById('config-text-wa').value;
    configData.texts.contact_email = document.getElementById('config-text-email').value;
    configData.texts.contact_address = document.getElementById('config-text-address').value;

    if (!configData.texts.home) configData.texts.home = {};
    ['services_pretitle', 'services_title', 'service1_title', 'service1_desc', 'service2_title', 'service2_desc', 'service3_title', 'service3_desc', 'service4_title', 'service4_desc',
     'process_pretitle', 'process_title', 'process1_title', 'process1_desc', 'process2_title', 'process2_desc', 'process3_title', 'process3_desc', 'process4_title', 'process4_desc',
     'diff_pretitle', 'diff_title', 'diff_desc', 'diff1_title', 'diff1_desc', 'diff2_title', 'diff2_desc', 'diff3_title', 'diff3_desc', 'diff4_title', 'diff4_desc', 'diff_item1', 'diff_item2', 'diff_item3',
     'testimonials_pretitle', 'testimonials_title', 'cta_title', 'cta_desc'].forEach(k => {
        const el = document.getElementById('config-home-' + k);
        if (el) configData.texts.home[k] = el.value;
    });

    if (!configData.texts.about) configData.texts.about = {};
    ['mission_vision_title', 'mission_vision_desc', 'team_pretitle', 'team_title', 'team_desc', 'map_iframe_url'].forEach(k => {
        const el = document.getElementById('config-about-' + k);
        if (el) configData.texts.about[k] = el.value;
    });

    if (!configData.texts.footer) configData.texts.footer = {};
    ['desc', 'copyright', 'legal'].forEach(k => {
        const el = document.getElementById('config-footer-' + k);
        if (el) configData.texts.footer[k] = el.value;
    });
    
    if (!configData.texts.aplicativos) configData.texts.aplicativos = {};
    configData.texts.aplicativos.title = document.getElementById('config-aplicativos-title').value;
    configData.texts.aplicativos.desc = document.getElementById('config-aplicativos-desc').value;
    
    const fields = [
        'empleado_link', 'empleado_hero_title', 'empleado_hero_desc', 'empleado_gateway_title', 'empleado_gateway_desc',
        'cliente_link', 'cliente_hero_title', 'cliente_hero_desc',
        'cliente_step1_title', 'cliente_step1_desc', 'cliente_step2_title', 'cliente_step2_desc', 'cliente_step3_title', 'cliente_step3_desc',
        'cliente_tools_title', 'cliente_tools_desc',
        'autorizacion_title', 'autorizacion_subtitle',
        'emailjs_public_key', 'emailjs_service_id', 'emailjs_template_autorizacion',
        'pqrs_title', 'pqrs_subtitle', 'emailjs_template_pqrs'
    ];
    fields.forEach(f => {
        const el = document.getElementById('config-aplicativos-' + f.replace(/_/g, '-'));
        if (el) configData.texts.aplicativos[f] = el.value;
    });

    if (!configData.texts.tienda) configData.texts.tienda = {};
    configData.texts.tienda.title = document.getElementById('config-tienda-title').value;
    configData.texts.tienda.desc = document.getElementById('config-tienda-desc').value;

    // Curso
    if (activeAdminCourseIndex !== -1) saveCurrentCourseConfig();

    // Collect team and gallery
    collectTeam();
    collectGallery();
    collectTestimonials();
    collectProducts();

    // Promises
    Promise.all([
        fetch('/api/data/config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(configData) }),
        fetch('/api/data/team', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(teamData) }),
        fetch('/api/data/gallery', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(galleryData) }),
        fetch('/api/data/testimonials', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(testimonialsData) }),
        fetch('/api/data/products', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(productsData) }),
        fetch('/api/data/courses', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(allAdminCourses) })
    ])
    .then(responses => Promise.all(responses.map(res => res.json())))
    .then(results => {
        const allSuccess = results.every(r => r.success);
        if (allSuccess) {
            showToast('Todo guardado con éxito ✓');
        } else {
            showToast('Hubo algunos errores al guardar ✗');
        }
    })
    .catch(err => {
        console.error('Error:', err);
        showToast('Error de conexión ✗');
    });
}

// ─── CURSO BENEFITS ────────────────────────────────────────
function renderCourseBenefits(benefits) {
    const container = document.getElementById('curso-benefits-admin');
    container.innerHTML = '';
    benefits.forEach((b, i) => {
        const row = document.createElement('div');
        row.className = 'flex items-center gap-3';
        row.innerHTML = `
            <span class="material-symbols-outlined text-primary text-lg shrink-0">check_circle</span>
            <input type="text" class="curso-benefit-input flex-1 bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm" value="${b}" />
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
    row.className = 'flex items-center gap-3';
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

// ─── USERS TABLE ────────────────────────────────────────────

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

function loadUsers() {
    fetch('/api/users?_t=' + Date.now())
        .then(res => res.json())
        .then(users => {
            const tbody = document.getElementById('users-table-body');
            if (users.length === 0) {
                tbody.innerHTML = '<tr><td colspan="3" class="py-8 text-center text-slate-500">No hay usuarios registrados aún.</td></tr>';
                return;
            }
            tbody.innerHTML = users.map(u => `
                <tr>
                    <td class="py-3 pr-4 font-semibold text-white">${u.nombre}</td>
                    <td class="py-3 pr-4 text-slate-400">${u.email}</td>
                    <td class="py-3 text-slate-500 text-xs">${u.fecha_registro ? new Date(u.fecha_registro).toLocaleString('es-CO') : '—'}</td>
                </tr>
            `).join('');
        })
        .catch(err => {
            console.error('Error loading users:', err);
            const tbody = document.getElementById('users-table-body');
            tbody.innerHTML = '<tr><td colspan="3" class="py-8 text-center text-red-400">Error al cargar usuarios.</td></tr>';
        });
}
