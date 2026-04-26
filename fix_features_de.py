with open('docs/de/features.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix redundant horizontal rules
content = content.replace('---\n\n---', '---')

with open('docs/de/features.md', 'w', encoding='utf-8') as f:
    f.write(content)
