import os
import re

def revert_app_py():
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Remove testimonials endpoint
    regex = re.compile(r"@app\.route\('/api/data/testimonials', methods=\['POST'\]\).*?return jsonify\(\{\"success\": True, \"message\": \"Testimonios guardados correctamente\"\}\)\n", re.DOTALL)
    content = regex.sub('', content)

    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)

def revert_dynamic_js():
    with open('dynamic.js', 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove testimonials section
    regex = re.compile(r"// 4\. Testimonials\s*if \(data\.testimonials && data\.testimonials\.length > 0\) \{.*?\n            \}\n", re.DOTALL)
    content = regex.sub('', content)
    
    with open('dynamic.js', 'w', encoding='utf-8') as f:
        f.write(content)

def revert_code_html():
    with open('code.html', 'r', encoding='utf-8') as f:
        content = f.read()

    original_testimonials = """<div class="grid grid-cols-1 md:grid-cols-3 gap-8">
                <div
                    class="p-8 rounded-3xl bg-background-light dark:bg-slate-900/50 border border-slate-200 dark:border-slate-800">
                    <div class="flex gap-1 text-primary mb-4">
                        <span class="material-symbols-outlined fill-1">star</span>
                        <span class="material-symbols-outlined fill-1">star</span>
                        <span class="material-symbols-outlined fill-1">star</span>
                        <span class="material-symbols-outlined fill-1">star</span>
                        <span class="material-symbols-outlined fill-1">star</span>
                    </div>
                    <p class="text-slate-600 dark:text-slate-400 italic mb-6 leading-relaxed">"Logré negociar una deuda
                        de 30 millones con solo 8 millones de pesos. Mi vida cambió totalmente gracias a MC Innovación."
                    </p>
                    <div class="flex items-center gap-4">
                        <div class="size-12 rounded-full bg-slate-200 overflow-hidden">
                            <img class="w-full h-full object-cover" data-alt="Portrait of a satisfied male client"
                                src="https://lh3.googleusercontent.com/aida-public/AB6AXuA81_t80eYHTFpGWzhGYq7T1ORukxaUmMnlOwldiwqeZ3TZs6RicSvr6sBLJE7LNbsPXvqVOnYFdmRzVkbMRwK8_ELgNC6LI2AW2PElnuwazHAf0epYo67iR8L4qQYWFvrSvBFE2UwTWcpLGeBOYGxE4SvcN0Y_wIFFR2jZllvHLTnScuYEMMtpngLDGxJCJxiFoQXXYRY6w54odANNxya4JZDRh50FE2W3vj2lZXBg0lybyrg5BzmB3qdz913gu80DPUUasCtkRFU" />
                        </div>
                        <div>
                            <p class="font-bold text-slate-900 dark:text-white">Carlos Rodriguez</p>
                            <p class="text-xs text-slate-500">Bogotá, D.C.</p>
                        </div>
                    </div>
                </div>
                <div
                    class="p-8 rounded-3xl bg-background-light dark:bg-slate-900/50 border border-slate-200 dark:border-slate-800">
                    <div class="flex gap-1 text-primary mb-4">
                        <span class="material-symbols-outlined fill-1">star</span>
                        <span class="material-symbols-outlined fill-1">star</span>
                        <span class="material-symbols-outlined fill-1">star</span>
                        <span class="material-symbols-outlined fill-1">star</span>
                        <span class="material-symbols-outlined fill-1">star</span>
                    </div>
                    <p class="text-slate-600 dark:text-slate-400 italic mb-6 leading-relaxed">"Excelente asesoría. Me
                        explicaron todo de forma clara y sin presiones. Hoy ya tengo mi score de crédito recuperado."
                    </p>
                    <div class="flex items-center gap-4">
                        <div class="size-12 rounded-full bg-slate-200 overflow-hidden">
                            <img class="w-full h-full object-cover" data-alt="Portrait of a satisfied female client"
                                src="https://lh3.googleusercontent.com/aida-public/AB6AXuBwsp1RjyZ602UOsIqat-ZsgKjJqqDI3Fws01kYywMW4drO4Hy9GtGwNcZx4MXN6CogGCrSsTqQJGxbow5kyjvxIQpTgLvYJ26klQXt4hFMWEWFiDNgnTNQvgaLhgCI5s-MiFyvovNCf-KEJHBHtIBEpfiE_dXBsDnp5VKkYGVH98fy80lvyOBFgHdwaj55t3RqmWfrE0EWKwG6nUSMS4Eqo-JCY6V0fqd7S4Vki2LN3bBvmZScQUYi360I8tAERTYgNfyBIm5AiCs" />
                        </div>
                        <div>
                            <p class="font-bold text-slate-900 dark:text-white">Ana María Holguín</p>
                            <p class="text-xs text-slate-500">Medellín</p>
                        </div>
                    </div>
                </div>
                <div
                    class="p-8 rounded-3xl bg-background-light dark:bg-slate-900/50 border border-slate-200 dark:border-slate-800">
                    <div class="flex gap-1 text-primary mb-4">
                        <span class="material-symbols-outlined fill-1">star</span>
                        <span class="material-symbols-outlined fill-1">star</span>
                        <span class="material-symbols-outlined fill-1">star</span>
                        <span class="material-symbols-outlined fill-1">star</span>
                        <span class="material-symbols-outlined fill-1">star</span>
                    </div>
                    <p class="text-slate-600 dark:text-slate-400 italic mb-6 leading-relaxed">"Su profesionalismo y
                        trato humano es lo que más destaco. Te hacen sentir apoyado en todo momento."</p>
                    <div class="flex items-center gap-4">
                        <div class="size-12 rounded-full bg-slate-200 overflow-hidden">
                            <img class="w-full h-full object-cover"
                                data-alt="Portrait of a satisfied professional client"
                                src="https://lh3.googleusercontent.com/aida-public/AB6AXuDHpXXJobIvptqls2iGvTAzctocBSb5DLMeQEQOMVcBvJJhH8V403kGqDWhbVulaiAybCN61QKzeIawRgo6ZsMww6ql-QjNbCmpXR2J2l_vaWbEdhIGgYTsdNNHJFZdJH20RDoWWODJ8xgIWsot2XYX6Wl9Uyq38Cbkj5ZLHZVDi1rYFYst8DeU4b_HEWXIGfSKfz4_zHMNU6p_O4ntO1L74rZKovvGZoxfxdiJJxZlrxdQvAIzIA5bVI3ut5tkD9f18LyGmE9q_fg" />
                        </div>
                        <div>
                            <p class="font-bold text-slate-900 dark:text-white">Jorge Varela</p>
                            <p class="text-xs text-slate-500">Cali</p>
                        </div>
                    </div>
                </div>
            </div>"""

    content = content.replace('<div id="testimonials-container" class="grid grid-cols-1 md:grid-cols-3 gap-8"></div>', original_testimonials)
    
    with open('code.html', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    revert_app_py()
    revert_dynamic_js()
    revert_code_html()
    print("Reverted simple files")
