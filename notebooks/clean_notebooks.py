import json
import os

notebooks = [
    "Data_Trip_ClickBus_Camada_Bronze.ipynb",
    "Data_Trip_ClickBus_Camada_Prata.ipynb",
    "Data_Trip_ClickBus_Camada_Ouro_EDA.ipynb",
    "Data_Trip_ClickBus_Camada_Ouro_Próximo_Trecho.ipynb",
    "Data_Trip_ClickBus_Camada_Ouro_Timing_é_tudo.ipynb",
    "Data_Trip_ClickBus_Camada_Ouro_Segmentação.ipynb",
    "Data_Trip_Modelo_final_ClickBus.ipynb"
    
]

for nome in notebooks:
    with open(nome, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        cell["outputs"] = []
        cell["execution_count"] = None
        cell.pop("id", None)

    nb.get("metadata", {}).pop("widgets", None)
    nb.get("metadata", {}).pop("accelerator", None)

    with open(nome, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"✅ {nome} — {os.path.getsize(nome) / 1024:.1f} KB")