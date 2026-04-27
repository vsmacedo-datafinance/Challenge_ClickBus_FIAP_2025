# Challenge ClickBus — FIAP 2025

Projeto de ciência de dados desenvolvido em parceria com a ClickBus,
plataforma líder em venda de passagens de ônibus no Brasil. Dados reais
de ~1,7 milhões de transações cobrindo set/2013 a abr/2024.

---

## Os Três Desafios

| # | Desafio | Abordagem | Métrica Principal |
|---|---------|-----------|-------------------|
| 1 | Segmentação de clientes | K-Means (4 clusters) | Separação visual via PCA
| 2 | Previsão de compra — próximos 30 dias | XGBoost + SMOTE
| 3 | Previsão do próximo trecho | LightGBM multiclasse

---

## Arquitetura Medallion

O pipeline segue a arquitetura em camadas Bronze → Prata → Ouro,
padrão de mercado para engenharia de dados em produção.

| Camada | Notebook | Responsabilidade |
|--------|----------|-----------------|
| Bronze | `01_Bronze.ipynb` | Ingestão e integridade do dado bruto |
| Prata  | `02_Prata.ipynb`  | Limpeza, decodificação e divisão temporal |
| Ouro   | `03_Ouro.ipynb`   | EDA, feature engineering e modelagem |

Funções reutilizáveis centralizadas em `src/utils.py`.

---

## Principais Decisões Técnicas

**Divisão temporal sem data leakage**
Dados divididos cronologicamente (treino até nov/2022, validação e teste
em sequência), evitando que o modelo aprenda com o futuro.

**Variável COVID como choque exógeno**
Classificação de cada transação em pré, durante e pós-pandemia usando
datas oficiais da OMS (mar/2020 – mai/2023), permitindo que os modelos
aprendam o comportamento distinto de cada período.

**Delta de 5 dias para feriados**
Compras realizadas nos 5 dias anteriores a feriados são sinalizadas como
potencialmente relacionadas ao evento, capturando intenção de viagem.

**Score de fidelidade por cliente**
Métrica inspirada em índice de concentração de portfólio: mede o percentual
do GMV do cliente concentrado em sua empresa preferida (0 = diversificado,
1 = exclusivo).

**SMOTE para desbalanceamento**
A taxa de compra em 30 dias é ~13%, exigindo oversampling sintético da
classe minoritária para o modelo não ignorar os positivos.

---

## Stack Técnica

Python · Pandas · XGBoost · LightGBM · Scikit-Learn · SMOTE · Parquet · Google Colab

---

## Estrutura do Repositório
