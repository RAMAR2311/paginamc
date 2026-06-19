import re

with open('admin.html', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Replace the HTML panel
new_panel = """
            <!-- ====== CURSO ====== -->
            <div id="panel-curso" class="hidden">
                <!-- VISTA LISTA DE CURSOS -->
                <div id="cursos-list-view">
                    <div class="flex justify-between items-center mb-6">
                        <h2 class="text-2xl font-extrabold flex items-center gap-2"><span class="material-symbols-outlined text-primary">school</span> Cursos</h2>
                        <button onclick="createNewCourse()" class="flex items-center gap-2 bg-primary text-background-dark px-6 py-2.5 rounded-xl font-bold hover:brightness-110 active:scale-95 transition-all">
                            <span class="material-symbols-outlined text-sm">add</span> Nuevo Curso
                        </button>
                    </div>
                    <div id="cursos-admin-list" class="grid grid-cols-1 md:grid-cols-2 gap-4"></div>
                </div>

                <!-- VISTA EDITOR DE CURSO -->
                <div id="curso-editor-view" class="hidden">
                    <div class="flex justify-between items-center mb-6">
                        <div class="flex items-center gap-4">
                            <button onclick="backToCoursesList()" class="bg-slate-800 text-slate-200 p-2 rounded-lg hover:bg-slate-700 transition-all"><span class="material-symbols-outlined">arrow_back</span></button>
                            <h2 class="text-2xl font-extrabold flex items-center gap-2"><span class="material-symbols-outlined text-primary">edit</span> Editar Curso</h2>
                        </div>
                        <button onclick="saveAll()" class="flex items-center gap-2 bg-primary text-background-dark px-6 py-2.5 rounded-xl font-bold hover:brightness-110 active:scale-95 transition-all">
                            <span class="material-symbols-outlined text-sm">save</span> Guardar Todo
                        </button>
                    </div>

                    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6">
                        <h3 class="text-lg font-bold mb-4">Configuración del Curso</h3>
                        <div class="space-y-4">
                            <div>
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Pretítulo</label>
                                <input type="text" id="config-curso-pretitle" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm" placeholder="Formación Premium" />
                            </div>
                            <div>
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Título del Curso</label>
                                <input type="text" id="config-curso-title" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm" placeholder="Eliminación de Reportes..." />
                            </div>
                            <div>
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Subtítulo / Descripción</label>
                                <textarea id="config-curso-subtitle" rows="3" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm" placeholder="Aprende paso a paso..."></textarea>
                            </div>
                            <div>
                                <label class="text-xs font-bold text-slate-400 mb-1 block">Foto del Curso</label>
                                <div class="flex items-center gap-4">
                                    <img id="config-curso-photo-preview" src="" class="w-24 h-16 object-cover rounded-lg bg-slate-800" />
                                    <input type="file" id="config-curso-photo-input" accept="image/*" class="hidden" />
                                    <button onclick="document.getElementById('config-curso-photo-input').click()" class="bg-slate-700 hover:bg-slate-600 px-4 py-2 rounded-lg text-sm font-bold">Subir Foto</button>
                                </div>
                            </div>
                            <div class="grid grid-cols-2 gap-4">
                                <div>
                                    <label class="text-xs font-bold text-slate-400 mb-1 block">Precio (COP)</label>
                                    <input type="number" id="config-curso-price" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm" placeholder="150000" />
                                </div>
                                <div>
                                    <label class="text-xs font-bold text-slate-400 mb-1 block">Llave Pública Wompi</label>
                                    <input type="text" id="config-curso-wompi-key" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm" placeholder="pub_prod_..." />
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Benefits -->
                    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 mt-6">
                        <div class="flex justify-between items-center mb-4">
                            <h3 class="text-lg font-bold">Beneficios del Curso</h3>
                            <button onclick="addCourseBenefit()" class="flex items-center gap-2 bg-slate-800 text-slate-200 px-4 py-2 rounded-lg font-bold text-sm hover:bg-slate-700 transition-all">
                                <span class="material-symbols-outlined text-sm">add</span> Agregar Beneficio
                            </button>
                        </div>
                        <div id="curso-benefits-admin" class="space-y-3"></div>
                    </div>

                    <!-- Modules & Lessons -->
                    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 mt-6">
                        <div class="flex justify-between items-center mb-4">
                            <h3 class="text-lg font-bold">Módulos y Lecciones</h3>
                            <button onclick="addModule()" class="flex items-center gap-2 bg-slate-800 text-slate-200 px-4 py-2 rounded-lg font-bold text-sm hover:bg-slate-700 transition-all">
                                <span class="material-symbols-outlined text-sm">add</span> Agregar Módulo
                            </button>
                        </div>
                        <div id="curso-modules-admin" class="space-y-6"></div>
                    </div>
                </div>
            </div>
"""

content = re.sub(r'<!-- ====== CURSO ====== -->.*?(?=<!-- ====== USUARIOS ====== -->)', new_panel, content, flags=re.DOTALL)

with open('admin.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("HTML Replaced!")
