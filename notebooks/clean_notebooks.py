import json
import os

notebooks = [
    "Data_Trip_ClickBus_Camada_Bronze.ipynb",
    "Data_Trip_ClickBus_Camada_Prata.ipynb",
    "Data_Trip_ClickBus_Camada_Ouro_EDA.ipynb",
    "Data_Trip_ClickBus_Camada_Ouro_Próximo_Trecho.ipynb",
    "Data_Trip_ClickBus_Camada_Ouro_Timing_é_tudo.ipynb",
    "Data_Trip_ClickBus_Camada_Ouro_Segmentação.ipynb",
    "Data_Trip_Modelo_final_ClickBus.ipynb",
]

for nome in notebooks:
    with open(nome, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        cell["outputs"] = [] if "outputs" in cell else cell.get("outputs", [])
        cell["execution_count"] = None
        # preserva metadata da célula mas remove widget state
        cell_meta = cell.get("metadata", {})
        cell_meta.pop("widgets", None)
        cell["metadata"] = cell_meta

    # remove só widgets e colab do metadata global, preserva nbformat
    nb_meta = nb.get("metadata", {})
    nb_meta.pop("widgets", None)
    nb_meta.pop("colab", None)
    nb_meta.pop("accelerator", None)
    nb["metadata"] = nb_meta

    with open(nome, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"✅ {nome} — {os.path.getsize(nome) / 1024:.1f} KB")