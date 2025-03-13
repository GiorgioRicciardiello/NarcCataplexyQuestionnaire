import re

# Read BibTeX entries from the provided text file
file_bibs = r'C:\Users\giorg\OneDrive - Fundacion Raices Italo Colombianas\projects\NarcCataplexyQuestionnaire\references\references_narco_paper.txt'
with open(file_bibs, 'r', encoding='utf-8') as file:
    bibtex_entries = file.read()

# Find all BibTeX entries in the input string
entries = re.findall(r'(@\w+\{.*?\n\})', bibtex_entries, re.DOTALL)

print(f"Total references found: {len(entries)}")

# Write each BibTeX entry to a separate file
matches = []
for entry in entries:
    match = re.match(r'@\w+\{(.*?),', entry)
    if match:
        matches.append(match.group(1))
        key = match.group(1).strip()
        filename = f"{key}.bib"
        with open(filename, "w", encoding='utf-8') as file:
            file.write(entry)
        print(f"Created {filename}")
print(f'total files {len(matches)}')