import os

with open('admin.html', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('onchange="updateCourseLesson', 'oninput="updateCourseLesson')
text = text.replace('onchange="updateCourseModule', 'oninput="updateCourseModule')

with open('admin.html', 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated admin.html events.')
