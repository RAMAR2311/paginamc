document.addEventListener('DOMContentLoaded', () => {
    fetch('/api/data?_t=' + Date.now())
        .then(res => res.json())
        .then(data => {
            if (!data.config) return;
            const config = data.config;
            const courses = data.courses || [];

            // If we are on curso.html, handle course selection and CTA linking
            const urlParams = new URLSearchParams(window.location.search);
            const courseId = urlParams.get('courseId');
            
            if (courses.length > 0 && config.texts && config.texts.curso) {
                // Determine which course to show (requested or first one)
                const selectedCourse = courseId ? courses.find(c => c.id === courseId) : courses[0];
                
                if (selectedCourse) {
                    if (courseId) {
                        // Override the config.texts.curso with the selected course if specifically requested
                        config.texts.curso.title = selectedCourse.title;
                        config.texts.curso.subtitle = selectedCourse.subtitle;
                        config.texts.curso.price = selectedCourse.price;
                        config.texts.curso.wompi_public_key = selectedCourse.wompi_public_key;
                        config.texts.curso.benefits = selectedCourse.benefits || [];
                        config.texts.curso.modules = selectedCourse.modules || [];
                        config.texts.curso.pretitle = selectedCourse.pretitle || 'Formación Premium';
                        config.texts.curso.photo = selectedCourse.photo;
                    }
                    
                    const ctaBtn = document.getElementById('curso-cta-btn');
                    if (ctaBtn) {
                        ctaBtn.href = '/comprar-curso?courseId=' + selectedCourse.id;
                    }
                }
            }

            // 1. Update CSS Variables for Tailwind Colors
            if (config.colors) {
                const root = document.documentElement;
                if (config.colors.primary) root.style.setProperty('--color-primary', config.colors.primary);
                if (config.colors.accentPink) root.style.setProperty('--color-accent-pink', config.colors.accentPink);
                if (config.colors.bgLight) root.style.setProperty('--color-bg-light', config.colors.bgLight);
                if (config.colors.bgDark) root.style.setProperty('--color-bg-dark', config.colors.bgDark);
            }

            // 2. Update Elements with data-edit attribute (inner content or src)
            document.querySelectorAll('[data-edit]').forEach(el => {
                const path = el.getAttribute('data-edit').split('.');
                let value = config;
                for (const key of path) {
                    if (value) value = value[key];
                }

                if (value !== undefined && value !== null) {
                    if (el.tagName === 'IMG' || el.tagName === 'IFRAME') {
                        el.src = value;
                    } else if (el.tagName === 'VIDEO' || el.tagName === 'SOURCE') {
                        el.src = value;
                        if (el.tagName === 'SOURCE' && el.parentElement.tagName === 'VIDEO') {
                            el.parentElement.load();
                        } else if (el.tagName === 'VIDEO') {
                            el.load();
                        }
                    } else {
                        // Use innerHTML so things like span with color works for hero_title
                        el.innerHTML = value;
                    }
                }
            });

            // 3. Update Elements with data-edit-href attribute for dynamic links
            document.querySelectorAll('[data-edit-href]').forEach(el => {
                const pathStr = el.getAttribute('data-edit-href');
                const matches = pathStr.match(/\{([^}]+)\}/g);
                let newHref = pathStr;
                
                if (matches) {
                    matches.forEach(match => {
                        const path = match.replace(/[{}]/g, '').split('.');
                        let value = config;
                        for (const key of path) {
                            if (value) value = value[key];
                        }
                        if (value !== undefined && value !== null) {
                            newHref = newHref.replace(match, value);
                        }
                    });
                }
                el.href = newHref;
            });

            // 4. Render Testimonials if container exists
            const testContainer = document.getElementById('testimonials-container');
            if (testContainer && data.testimonials) {
                renderTestimonialsList(testContainer, data.testimonials);
                initTestimonialsCarousel();
            }

            // 4.5 Render Services if container exists
            const servicesContainer = document.getElementById('services-container');
            if (servicesContainer && data.services) {
                renderServicesList(servicesContainer, data.services);
                initServicesCarousel();
            }

            // 5. Render Store Products if container exists
            const productsContainer = document.getElementById('dynamic-products-container');
            if (productsContainer && data.products) {
                renderStoreProductsList(productsContainer, data.products);
                document.dispatchEvent(new CustomEvent('productsRendered'));
            }

            // 6. Render Course Section if it exists
            const cursoConfig = config.texts?.curso;
            if (cursoConfig) {
                // Render benefits list
                const benefitsList = document.getElementById('curso-benefits-list');
                if (benefitsList && cursoConfig.benefits && Array.isArray(cursoConfig.benefits)) {
                    benefitsList.innerHTML = cursoConfig.benefits.map(b => `
                        <li class="flex items-start gap-3 text-slate-700 dark:text-slate-300">
                            <span class="material-symbols-outlined text-primary text-xl mt-0.5 shrink-0">check_circle</span>
                            <span>${b}</span>
                        </li>
                    `).join('');
                }

                // Render price
                const priceDisplay = document.getElementById('curso-price-display');
                if (priceDisplay && cursoConfig.price) {
                    priceDisplay.textContent = '$' + parseInt(cursoConfig.price).toLocaleString('es-CO');
                }
            }

            // Dispatch global event for other scripts to access the config
            document.dispatchEvent(new CustomEvent('configLoaded', { detail: config }));
        })
        .catch(err => console.error('Error loading dynamic content:', err));
});

function renderTestimonialsList(container, list) {
    container.innerHTML = list.map(t => {
        let starsHtml = '';
        const starCount = parseInt(t.stars) || 5;
        for (let i = 0; i < starCount; i++) {
            starsHtml += '<span class="material-symbols-outlined fill-1">star</span>';
        }
        const defaultPhoto = 'https://lh3.googleusercontent.com/aida-public/AB6AXu81_t80eYHTFpGWzhGYq7T1ORukxaUmMnlOwldiwqeZ3TZs6RicSvr6sBLJE7LNbsPXvqVOnYFdmRzVkbMRwK8_ELgNC6LI2AW2PElnuwazHAf0epYo67iR8L4qQYWFvrSvBFE2UwTWcpLGeBOYGxE4SvcN0Y_wIFFR2jZllvHLTnScuYEMMtpngLDGxJCJxiFoQXXYRY6w54odANNxya4JZDRh50FE2W3vj2lZXBg0lybyrg5BzmB3qdz913gu80DPUUasCtkRFU';
        return `
            <div class="min-w-full md:min-w-[calc(33.333%-1.333rem)] snap-start p-8 rounded-3xl bg-background-light dark:bg-slate-900/50 border border-slate-200 dark:border-slate-800 flex flex-col justify-between">
                <div>
                    <div class="flex gap-1 text-primary mb-4">
                        ${starsHtml}
                    </div>
                    <p class="text-slate-600 dark:text-slate-400 italic mb-6 leading-relaxed">"${t.text}"</p>
                </div>
                <div class="flex items-center gap-4 mt-auto">
                    <div class="size-12 rounded-full bg-slate-200 overflow-hidden shrink-0">
                        <img class="w-full h-full object-cover" src="${t.photo || defaultPhoto}" alt="${t.name}" onerror="this.src='${defaultPhoto}'" />
                    </div>
                    <div>
                        <p class="font-bold text-slate-900 dark:text-white">${t.name}</p>
                        <p class="text-xs text-slate-500">${t.location}</p>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function initTestimonialsCarousel() {
    const container = document.getElementById('testimonials-container');
    const prevBtn = document.getElementById('prev-testimonials');
    const nextBtn = document.getElementById('next-testimonials');

    if (!container || !prevBtn || !nextBtn) return;

    const getScrollAmount = () => {
        const firstItem = container.querySelector('div');
        if (!firstItem) return 0;
        return firstItem.offsetWidth + 32; // width + gap
    };

    nextBtn.addEventListener('click', () => {
        container.scrollBy({ left: getScrollAmount(), behavior: 'smooth' });
    });

    prevBtn.addEventListener('click', () => {
        container.scrollBy({ left: -getScrollAmount(), behavior: 'smooth' });
    });
}

function renderServicesList(container, list) {
    container.innerHTML = list.map(s => {
        const iconName = s.icon || 'handshake';
        return `
            <div class="min-w-[260px] max-w-[280px] snap-start shrink-0 group p-6 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 hover:border-primary dark:hover:border-primary transition-all hover:shadow-2xl hover:shadow-primary/5 flex flex-col">
                <div class="size-12 bg-primary rounded-xl flex items-center justify-center text-background-dark mb-4 group-hover:rotate-6 transition-transform">
                    <span class="material-symbols-outlined text-2xl">${iconName}</span>
                </div>
                <h4 class="text-lg font-bold mb-2 leading-tight">${s.title}</h4>
                <p class="text-slate-600 dark:text-slate-400 text-xs leading-relaxed mt-auto">${s.desc}</p>
            </div>
        `;
    }).join('');
}

function initServicesCarousel() {
    const container = document.getElementById('services-container');
    const prevBtn = document.getElementById('prev-services');
    const nextBtn = document.getElementById('next-services');

    if (!container || !prevBtn || !nextBtn) return;

    const getScrollAmount = () => {
        const firstItem = container.querySelector('div');
        if (!firstItem) return 0;
        return firstItem.offsetWidth + 32; // width + gap
    };

    nextBtn.addEventListener('click', () => {
        container.scrollBy({ left: getScrollAmount(), behavior: 'smooth' });
    });

    prevBtn.addEventListener('click', () => {
        container.scrollBy({ left: -getScrollAmount(), behavior: 'smooth' });
    });
}

function renderStoreProductsList(container, list) {
    container.innerHTML = list.map(p => {
        let tagHtml = '';
        if (p.tag) {
            let tagColorClass = p.tag === 'Recomendado' ? 'bg-accent-pink text-white shadow-accent-pink/30' : 'bg-primary text-background-dark shadow-primary/30';
            tagHtml = `
                <div class="absolute top-5 left-5 z-10">
                    <span class="${tagColorClass} text-[10px] font-extrabold px-3 py-1.5 rounded-full uppercase tracking-widest shadow-lg">${p.tag}</span>
                </div>
            `;
        }
        
        let priceFormatted = parseInt(p.price).toLocaleString('es-CO');
        let iconName = p.icon || 'analytics';
        
        let mediaHtml = '';
        if (p.photo) {
            mediaHtml = `<img src="${p.photo}" class="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105" alt="${p.title}" />`;
        } else {
            mediaHtml = `<span class="material-symbols-outlined text-6xl text-slate-300 dark:text-slate-600 group-hover:scale-110 group-hover:text-primary transition-all duration-500 relative z-10 drop-shadow-sm">${iconName}</span>`;
        }
        
        return `
            <div class="producto-card group bg-white/60 dark:bg-slate-800/30 backdrop-blur-xl rounded-[2rem] border border-white/60 dark:border-slate-700/50 shadow-[0_8px_30px_rgb(0,0,0,0.04)] dark:shadow-[0_8px_30px_rgb(0,0,0,0.15)] hover:shadow-[0_20px_40px_rgba(236,203,19,0.1)] hover:-translate-y-1 transition-all duration-300 overflow-hidden flex flex-col relative text-left" data-categoria="${p.category}">
                ${tagHtml}
                <div class="h-48 bg-gradient-to-br from-slate-100 to-slate-200 dark:from-slate-800 dark:to-slate-900 flex items-center justify-center relative overflow-hidden">
                    <div class="absolute inset-0 bg-primary/0 group-hover:bg-primary/5 transition-colors duration-500"></div>
                    <div class="absolute w-32 h-32 bg-white/40 dark:bg-black/20 rounded-full blur-xl group-hover:scale-150 transition-transform duration-700"></div>
                    ${mediaHtml}
                </div>
                <div class="p-8 flex-1 flex flex-col">
                    <h3 class="producto-titulo text-xl font-bold mb-3 text-slate-900 dark:text-white group-hover:text-primary transition-colors leading-snug">${p.title}</h3>
                    <p class="producto-desc text-slate-500 dark:text-slate-400 text-sm mb-8 line-clamp-3 leading-relaxed">${p.desc}</p>
                    <div class="mt-auto">
                        <p class="text-3xl font-extrabold mb-5 text-slate-900 dark:text-white tracking-tight">$${priceFormatted} <span class="text-sm font-semibold text-slate-400 tracking-normal">COP</span></p>
                        <button class="w-full bg-primary text-background-dark font-bold py-3.5 rounded-xl border border-transparent hover:-translate-y-1 hover:shadow-[0_8px_20px_rgba(236,203,19,0.3)] hover:brightness-110 active:scale-95 transition-all duration-300 flex items-center justify-center gap-2 shadow-sm">
                            <span class="material-symbols-outlined text-[20px]">add_shopping_cart</span> Añadir al Carrito
                        </button>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// ==========================================
// SEGURIDAD FRONTEND
// ==========================================
// 1. Deshabilitar el clic derecho en toda la página
document.addEventListener('contextmenu', event => event.preventDefault());

// 2. Deshabilitar teclas específicas
document.addEventListener('keydown', function(e) {
    // Bloquear F12
    if (e.key === 'F12' || e.keyCode === 123) {
        e.preventDefault();
    }
    // Bloquear Ctrl+Shift+I (Inspeccionar en Chrome)
    if (e.ctrlKey && e.shiftKey && (e.key === 'I' || e.key === 'i' || e.keyCode === 73)) {
        e.preventDefault();
    }
    // Bloquear Ctrl+Shift+J (Consola en Chrome)
    if (e.ctrlKey && e.shiftKey && (e.key === 'J' || e.key === 'j' || e.keyCode === 74)) {
        e.preventDefault();
    }
    // Bloquear Ctrl+Shift+C (Inspeccionar Elemento)
    if (e.ctrlKey && e.shiftKey && (e.key === 'C' || e.key === 'c' || e.keyCode === 67)) {
        e.preventDefault();
    }
    // Bloquear Ctrl+U (Ver código fuente de la página)
    if (e.ctrlKey && (e.key === 'U' || e.key === 'u' || e.keyCode === 85)) {
        e.preventDefault();
    }
});
