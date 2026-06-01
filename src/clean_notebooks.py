# clean_notebooks.py
import json
import glob
import os

notebooks = glob.glob('notebooks/*.ipynb')

for path in notebooks:
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Remove metadata do Colab, mantém só o essencial
    nb['metadata'] = {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.12.7'
        }
    }

    # Limpa outputs e execution_count de todas as células
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    size_kb = os.path.getsize(path) / 1024
    print(f'Limpo: {path} ({size_kb:.1f} KB)')

print('\nPronto. Agora commita e faz push.')