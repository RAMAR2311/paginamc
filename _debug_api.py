import urllib.request, json

resp = urllib.request.urlopen('http://localhost:5000/api/data')
data = json.loads(resp.read().decode('utf-8'))
courses = data.get('courses', [])
print(f'Number of courses from API: {len(courses)}')
for c in courses:
    cid = c.get("id", "no-id")
    title = c.get("title", "no-title")
    print(f'  - {cid}: {title}')
