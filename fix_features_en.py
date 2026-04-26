with open('docs/en/features.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip_next = False
for i in range(len(lines)):
    if "LLM Client automatically detects which LLM provider to use based on available API keys." in lines[i]:
        if i + 2 < len(lines) and "LLM Client automatically detects which LLM provider to use based on available API keys:" in lines[i+2]:
            # This is the redundancy
            continue
    new_lines.append(lines[i])

with open('docs/en/features.md', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
