import json

with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

courses = data.get('courses', [])
for c in courses:
    print("Course:", c.get("title"))
    for m in c.get('modules', []):
        print("  Module:", m.get("title"))
        for l in m.get('lessons', []):
            vurl = l.get('video_url', '')
            title = l.get("title", "?")
            if vurl:
                print(f"    Lesson: {title} -> {vurl}")
            else:
                print(f"    Lesson: {title} -> NO VIDEO URL")
