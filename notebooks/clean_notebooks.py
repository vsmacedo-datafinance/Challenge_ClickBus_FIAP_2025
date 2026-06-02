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
        if cell["cell_type"] == "code":  # só células de código!
            cell["outputs"] = []
            cell["execution_count"] = None

    nb_meta = nb.get("metadata", {})
    nb_meta.pop("widgets", None)
    nb_meta.pop("colab", None)
    nb_meta.pop("accelerator", None)
    nb["metadata"] = nb_meta

    with open(nome, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"✅ {nome} — {os.path.getsize(nome) / 1024:.1f} KB")