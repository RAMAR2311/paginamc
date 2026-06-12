import re
import os

CODE_FILE = 'code.html'
ABOUT_FILE = 'nosotros.html'

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

# Update code.html
code_content = read_file(CODE_FILE)

# 1. Services Section
code_content = code_content.replace('>Nuestros Servicios Premium', ' data-edit="texts.home.services_pretitle">Nuestros Servicios Premium')
code_content = code_content.replace('>Soluciones integrales a tu medida', ' data-edit="texts.home.services_title">Soluciones integrales a tu medida')

code_content = code_content.replace('>Negociación de deudas</h4>', ' data-edit="texts.home.service1_title">Negociación de deudas</h4>')
code_content = code_content.replace('>Reducimos tus saldos', ' data-edit="texts.home.service1_desc">Reducimos tus saldos')

code_content = code_content.replace('>Reparación de historial</h4>', ' data-edit="texts.home.service2_title">Reparación de historial</h4>')
code_content = code_content.replace('>Mejoramos tu score en', ' data-edit="texts.home.service2_desc">Mejoramos tu score en')

code_content = code_content.replace('>Asesoría personalizada</h4>', ' data-edit="texts.home.service3_title">Asesoría personalizada</h4>')
code_content = code_content.replace('>Creamos un plan estratégico', ' data-edit="texts.home.service3_desc">Creamos un plan estratégico')

code_content = code_content.replace('>Educación financiera</h4>', ' data-edit="texts.home.service4_title">Educación financiera</h4>')
code_content = code_content.replace('>Te brindamos las herramientas', ' data-edit="texts.home.service4_desc">Te brindamos las herramientas')

# 2. Process Section
code_content = code_content.replace('>Metodología</h2>', ' data-edit="texts.home.process_pretitle">Metodología</h2>')
code_content = code_content.replace('>Tu camino hacia la', ' data-edit="texts.home.process_title">Tu camino hacia la')

code_content = code_content.replace('>Diagnóstico</h5>', ' data-edit="texts.home.process1_title">Diagnóstico</h5>')
code_content = code_content.replace('>Evaluamos tu situación actual, nivel de deuda', ' data-edit="texts.home.process1_desc">Evaluamos tu situación actual, nivel de deuda')

code_content = code_content.replace('>Análisis</h5>', ' data-edit="texts.home.process2_title">Análisis</h5>')
code_content = code_content.replace('>Identificamos oportunidades de ahorro y los', ' data-edit="texts.home.process2_desc">Identificamos oportunidades de ahorro y los')

code_content = code_content.replace('>Estrategia</h5>', ' data-edit="texts.home.process3_title">Estrategia</h5>')
code_content = code_content.replace('>Ejecutamos el plan de acción, logrando los', ' data-edit="texts.home.process3_desc">Ejecutamos el plan de acción, logrando los')

code_content = code_content.replace('>Recuperación</h5>', ' data-edit="texts.home.process4_title">Recuperación</h5>')
code_content = code_content.replace('>Saneamos tu historial y te entregamos un plan', ' data-edit="texts.home.process4_desc">Saneamos tu historial y te entregamos un plan')

# 3. Diferenciales
code_content = code_content.replace('>Diferenciales</h2>', ' data-edit="texts.home.diff_pretitle">Diferenciales</h2>')
code_content = code_content.replace('>¿Por qué confiar en MC', ' data-edit="texts.home.diff_title">¿Por qué confiar en MC')
code_content = code_content.replace('>\n                    Entendemos que detrás de cada deuda hay una persona y una familia buscando paz. No somos una agencia\n                    de cobro, somos tu defensa frente al sistema financiero.\n                </p>', ' data-edit="texts.home.diff_desc">\n                    Entendemos que detrás de cada deuda hay una persona y una familia buscando paz. No somos una agencia\n                    de cobro, somos tu defensa frente al sistema financiero.\n                </p>')

code_content = code_content.replace('>Transparencia</h6>', ' data-edit="texts.home.diff1_title">Transparencia</h6>')
code_content = code_content.replace('>Sin letras pequeñas ni cobros ocultos.</p>', ' data-edit="texts.home.diff1_desc">Sin letras pequeñas ni cobros ocultos.</p>')

code_content = code_content.replace('>Confidencialidad</h6>', ' data-edit="texts.home.diff2_title">Confidencialidad</h6>')
code_content = code_content.replace('>Tus datos están protegidos legalmente.</p>', ' data-edit="texts.home.diff2_desc">Tus datos están protegidos legalmente.</p>')

code_content = code_content.replace('>Profesionalismo</h6>', ' data-edit="texts.home.diff3_title">Profesionalismo</h6>')
code_content = code_content.replace('>Consultores certificados por ley.</p>', ' data-edit="texts.home.diff3_desc">Consultores certificados por ley.</p>')

code_content = code_content.replace('>Personalización</h6>', ' data-edit="texts.home.diff4_title">Personalización</h6>')
code_content = code_content.replace('>Nadie tiene el mismo plan que tú.</p>', ' data-edit="texts.home.diff4_desc">Nadie tiene el mismo plan que tú.</p>')

code_content = code_content.replace('Acompañamiento legal constante', '<span data-edit="texts.home.diff_item1">Acompañamiento legal constante</span>')
code_content = code_content.replace('Resultados garantizados por contrato', '<span data-edit="texts.home.diff_item2">Resultados garantizados por contrato</span>')
code_content = code_content.replace('Planes de pago flexibles', '<span data-edit="texts.home.diff_item3">Planes de pago flexibles</span>')

# 4. Testimonios
code_content = code_content.replace('>Testimonios</h2>', ' data-edit="texts.home.testimonials_pretitle">Testimonios</h2>')
code_content = code_content.replace('>Lo que dicen nuestros clientes</h3>', ' data-edit="texts.home.testimonials_title">Lo que dicen nuestros clientes</h3>')

# 5. CTA Final
code_content = code_content.replace('>¿Listo para\n                        volver a dormir tranquilo?</h2>', ' data-edit="texts.home.cta_title">¿Listo para\n                        volver a dormir tranquilo?</h2>')
code_content = code_content.replace('>Únete a los miles de colombianos que ya\n                        recuperaron su libertad financiera con nuestra ayuda.</p>', ' data-edit="texts.home.cta_desc">Únete a los miles de colombianos que ya\n                        recuperaron su libertad financiera con nuestra ayuda.</p>')

write_file(CODE_FILE, code_content)
print("Updated code.html")

# Update nosotros.html
about_content = read_file(ABOUT_FILE)
about_content = about_content.replace('>\n                Nuestra <span class="text-primary italic">Misión</span> y <span', ' data-edit="texts.about.mission_vision_title">\n                Nuestra <span class="text-primary italic">Misión</span> y <span')
about_content = about_content.replace('>\n                Nuestra misión es la defensa integral de tus derechos financieros y personales para que recuperes tu libertad crediticia.', ' data-edit="texts.about.mission_vision_desc">\n                Nuestra misión es la defensa integral de tus derechos financieros y personales para que recuperes tu libertad crediticia.')

about_content = about_content.replace('>Nuestro Equipo</h2>', ' data-edit="texts.about.team_pretitle">Nuestro Equipo</h2>')
about_content = about_content.replace('>Conoce a tus defensores financieros</h3>', ' data-edit="texts.about.team_title">Conoce a tus defensores financieros</h3>')
about_content = about_content.replace('>Un grupo de 7 expertos comerciales\n                    y legales firmemente constituidos como tu aliado estratégico.', ' data-edit="texts.about.team_desc">Un grupo de 7 expertos comerciales\n                    y legales firmemente constituidos como tu aliado estratégico.')

write_file(ABOUT_FILE, about_content)
print("Updated nosotros.html")

# Update footers in all html files
import glob
for file in glob.glob('*.html'):
    if file == 'admin.html': continue
    content = read_file(file)
    content = content.replace('>Expertos en asesoría financiera y soluciones de deuda en todo el\n                        territorio colombiano.</p>', ' data-edit="texts.footer.desc">Expertos en asesoría financiera y soluciones de deuda en todo el\n                        territorio colombiano.</p>')
    content = content.replace('>© 2024 MC Innovación Financiera. Todos los derechos reservados.</p>', ' data-edit="texts.footer.copyright">© 2024 MC Innovación Financiera. Todos los derechos reservados.</p>')
    content = content.replace('>*Toda nuestra gestión se realiza bajo el estricto marco legal de la Ley 1266 de 2008 (Ley de Habeas Data) y la Ley 1581 de 2012.\n                </div>', ' data-edit="texts.footer.legal">*Toda nuestra gestión se realiza bajo el estricto marco legal de la Ley 1266 de 2008 (Ley de Habeas Data) y la Ley 1581 de 2012.\n                </div>')
    write_file(file, content)
print("Updated footers")
