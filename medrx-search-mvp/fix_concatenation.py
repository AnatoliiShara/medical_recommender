import re

# Читаємо файл
with open('src/eda/comprehensive_medical_eda.py', 'r') as f:
    content = f.read()

# Знаходимо та замінюємо проблемну частину
old_pattern = r"concatenated = ' \[SEP\] '\.join\(concat_parts\)"
new_code = '''if len(concat_parts) > 1:
                    concatenated = concat_parts[0].astype(str)
                    for part in concat_parts[1:]:
                        concatenated = concatenated + ' [SEP] ' + part.astype(str)
                else:
                    concatenated = concat_parts[0].astype(str)'''

content = re.sub(old_pattern, new_code, content)

# Записуємо виправлений файл
with open('src/eda/comprehensive_medical_eda.py', 'w') as f:
    f.write(content)
    
print("Файл виправлено!")
