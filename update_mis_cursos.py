with open('mis-cursos.html', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add HTML container for available courses
html_to_add = """
            <div id="available-courses-section" class="mt-16">
                <h2 class="text-3xl font-extrabold text-background-dark dark:text-white mb-8">Cursos Disponibles</h2>
                <div id="available-courses-container" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                    <!-- Dynamic -->
                </div>
            </div>
        </div>
    </main>
"""
content = content.replace('        </div>\n    </main>', html_to_add)

# 2. Update JS to populate available courses
js_old = """            // Filtrar y renderizar solo los cursos comprados
            const purchasedCourses = allCourses.filter(c => purchasedIds.includes(c.id));
            
            if (purchasedCourses.length === 0) {
                noCourses.classList.remove('hidden');
            } else {"""

js_new = """            // Filtrar y renderizar solo los cursos comprados
            const purchasedCourses = allCourses.filter(c => purchasedIds.includes(c.id));
            const unpurchasedCourses = allCourses.filter(c => !purchasedIds.includes(c.id));
            
            // Render unpurchased courses
            const availableContainer = document.getElementById('available-courses-container');
            const availableSection = document.getElementById('available-courses-section');
            if(availableContainer) {
                if(unpurchasedCourses.length === 0) {
                    availableSection.style.display = 'none';
                } else {
                    unpurchasedCourses.forEach(c => {
                        const photo = c.photo || 'https://images.unsplash.com/photo-1556761175-b413da4baf72?auto=format&fit=crop&q=80&w=800';
                        availableContainer.innerHTML += `
                            <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-3xl overflow-hidden shadow-lg hover:shadow-xl hover:shadow-primary/5 transition-all flex flex-col group">
                                <div class="aspect-video relative overflow-hidden bg-slate-100 dark:bg-slate-800">
                                    <img src="${photo}" alt="${c.title}" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700">
                                </div>
                                <div class="p-6 flex flex-col flex-grow">
                                    <h3 class="text-xl font-extrabold text-slate-900 dark:text-white mb-2 line-clamp-2">${c.title}</h3>
                                    <p class="text-slate-500 dark:text-slate-400 text-sm mb-6 flex-grow line-clamp-2">${c.subtitle || 'Aprende y domina este tema con nuestros expertos.'}</p>
                                    
                                    <a href="curso.html?courseId=${c.id}" class="w-full bg-primary text-background-dark py-3 rounded-xl font-bold flex items-center justify-center gap-2 hover:brightness-110 transition-all">
                                        <span class="material-symbols-outlined">info</span> Ver Curso
                                    </a>
                                </div>
                            </div>
                        `;
                    });
                }
            }
            
            if (purchasedCourses.length === 0) {
                noCourses.classList.remove('hidden');
            } else {"""

content = content.replace(js_old, js_new)

with open('mis-cursos.html', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated mis-cursos.html!")
