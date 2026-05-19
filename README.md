# ClickBus — Decodificando o Comportamento do Viajante

**FIAP Innovation Challenge 2025 · Refatoração v2 · Portfólio de Data Science** ***

## Contexto

A ClickBus é a maior plataforma de venda de passagens rodoviárias do Brasil, atuando em um mercado estimado em 170 milhões de tickets/ano e R$ 20 bilhões em GMV, ainda majoritariamente offline. Este projeto usa dados reais anonimizados do Innovation Challenge FIAP 2025 para entender o comportamento de compra e apoiar decisões de Growth.

Foco desta versão: **arquitetura de dados + EDA + segmentação + previsão de recompra (Timing é Tudo)**.

*(O modelo de **próximo trecho** está planejado e em desenvolvimento).*

---

## Versão refatorada (v2) vs. versão original

A solução original foi entregue no 1º ano da graduação, com a maior parte da lógica centralizada em um único notebook e risco de data leakage.

A refatoração v2, desenvolvida no 2º ano, reorganiza o projeto em:

* **Arquitetura Medalhão:** Bronze (ingestão); Prata (limpeza); Ouro (EDA e modelos).
* **Código modular:** funções de regra de negócio em `src/utils.py`.
* **Prevenção de leakage por design:** toda estatística (popularidade de trechos, scores de fidelidade, proporções de comportamento) é calculada apenas em `df_treino` e aplicada em validação/teste via `.map()`.

Os modelos continuam os mesmos (K-Means na segmentação), mas a disciplina de dados e a leitura econômica evoluíram, transformando o trabalho em um verdadeiro Produto de Dados.

---

## Diferenciais analíticos

* **Visão de Economia aplicada a dados**
* `periodo_covid` (pré / durante / pós) tratado como choque exógeno estruturante.
* `score_fidelidade` como proxy de market share individual (GMV da empresa favorita / GMV total do cliente).
* Leitura de trade-offs de Growth: custo de campanha vs. valor esperado de cliente.


* **Framework RFM estendido em nível de cliente**
* Monetary: `gmv_total`, `gmv_total_log`.
* Frequency: `total_compras`, `total_compras_log`.
* Recency: `recencia_dias`, `tempo_vida_dias`.
* Comportamento: `n_destinos_distintos`, `prop_feriado`, `prop_ida_volta`, `prop_fim_semana`.
* Fidelidade: `score_fidelidade`.


* **Rigor estatístico na EDA**
* Transformação `log1p` em variáveis altamente assimétricas.
* Correlações Pearson e Spearman para relações lineares/monotônicas.
* VIF para diagnóstico de multicolinearidade e remoção de variáveis redundantes com justificativa.



---

## Arquitetura e Notebooks

* **01_Camada_Bronze.ipynb — Ingestão e integridade** - Lê o CSV bruto (~1,7M linhas, 2013–2024).
* Valida schema, nulos e intervalo de datas.
* Salva `clickbus_bronze.parquet` como fonte imutável.


* **02_Camada_Prata.ipynb — Limpeza e preparação** - Decodifica hashes de cidades, empresas e clientes.
* Remove registros inconsistentes (GMV ≤ 0, origem = destino, dados de ida ausentes) com log de impacto.
* Cria features determinísticas (`e_feriado`, `compra_ate_5_dias_feriado`, `trecho_ida`, `empresa_ida`).
* Realiza split temporal out-of-time: treino; validação; teste (sem embaralhar).
* Define `periodo_covid` (pré / durante / pós) para capturar o choque da pandemia.


* **03_Camada_Ouro_EDA.ipynb — EDA + df_cliente** - EDA apenas em `df_treino` para evitar leakage.
* Análises em nível transação: sazonalidade, concentração de rotas, participação de empresas, série de GMV com banda COVID.
* Construção de `df_cliente` agregando RFM + comportamento + fidelidade.
* Saídas: `df_cliente.parquet` e `df_cluster.parquet` (preparado para K-Means).


* **04_Camada_Ouro_Segmentacao.ipynb — K-Means (5 clusters)** - Carrega `df_cluster.parquet` (id_cliente + 11 features numéricas) e aplica `StandardScaler`.
* Testa k de 3 a 10 com método do cotovelo e Silhouette Score.
* Fixa **k = 5** como compromisso entre qualidade de separação e clareza de personas para Growth.
* Salva `df_cliente_clusterizado.parquet` (id_cliente + features + cluster + nome_cluster).


* **05_Camada_Ouro_Timing_e_Tudo.ipynb — XGBoost (Previsão de Recompra a 30 dias)**
* Enriquece `df_cliente` com features comportamentais calculadas sobre `df_treino`: `sazonalidade_score`, `prop_ferias`, `mes_ultima_compra`, `intervalo_medio_dias`, `dias_ate_proximo_feriado`.
* Constrói o target binário a partir de `df_val` filtrado a 30 dias após o corte temporal. O uso da janela de 30 dias previne a inflação irreal dos positivos de ~4% para ~40%.
* Aplica One-Hot Encoding no cluster e split estratificado por cliente.
* Compara três estratégias de balanceamento via `ImbPipeline` com sampler dentro do CV (sem leakage): Undersampling, SMOTE e `scale_pos_weight`.
* Seleciona o modelo vencedor por AUC-PR, que se mostra mais honesta que AUC-ROC em dados fortemente desbalanceados (24.6:1).
* Calibra o threshold por máximo F1 na curva Precision-Recall para maximizar o ROI de campanhas.
* Interpreta as previsões via SHAP (TreeExplainer), evidenciando a importância global e a direção do efeito por feature.
* Saídas: `ouro/xgboost_30d.pkl` · `ouro/df_modelo_classificacao.parquet`.



---

## Modelos Aplicados

### Segmentação — K-Means (k = 5)

Modelo: K-Means sobre as features numéricas de cliente (logs de valor/frequência/recência/diversidade, proporções de feriado/fim de semana/ida-volta, ticket médio, score de fidelidade), após padronização.

**Personas identificadas**

| Cluster | Nome | Características principais | Uso de negócio |
| --- | --- | --- | --- |
| 0 | Recorrente Explorer | Valor e frequência acima da média, mais destinos distintos | Cross-sell de rotas, recomendações personalizadas |
| 1 | Heavy User Volátil | Maior GMV e frequência da base, baixa fidelidade a empresa | Programa de fidelidade, captura de share of wallet |
| 2 | Famílias / Grupo | Ticket médio alto, forte uso de ida-volta, multi-tickets | Combos, descontos progressivos, comunicação para grupos |
| 3 | One-shot de Feriado | Compra única concentrada em feriados | Campanhas sazonais 30–45 dias antes de feriados prolongados |
| 4 | One-shot Regular | Compra única fora de feriado, fidelidade ligeiramente maior | Campanhas de reativação e incentivo à segunda viagem |

### Previsão de Recompra — XGBoost

Modelo de propensão treinado sobre as features RFM estendidas por cliente. O target foi construído verificando quem aparece na base de validação dentro de uma janela de 30 dias após o corte do treino.

* **O Desafio do Desbalanceamento:** A taxa real de positivos é de ~4% (um Ratio de 24.6:1), tornando-se um problema genuinamente difícil onde o modelo aleatório teria AUC-PR de ~0.04. A solução vencedora penalizou os erros nativamente via `scale_pos_weight` no estimador.
* **Calibração de Threshold:** Buscando o "Sweet Spot" de máximo F1 na curva Precision-Recall, a régua de probabilidade foi elevada para 79%. Isso equilibra o custo de não ativar um comprador real (falso negativo) com o custo de ativar falsos positivos.
* **Resultados e Multiplicador de Receita:** Com esse rigor, o modelo atingiu uma Precisão de 27.34% (quase 6x a taxa natural de conversão) mantendo um Recall saudável, garantindo um ROI agressivo para a equipe de Growth nas ações de retargeting.
* **Interpretabilidade (SHAP):** As variáveis rainhas da conversão provaram ser a **Recência** (dias desde a última compra) e o **Ritmo de Compra** (`intervalo_medio_dias`). Clientes com recência baixa são os principais motores da probabilidade predita, alinhando-se perfeitamente com a teoria de Customer Lifetime Value.

---

## Como executar

1. Preparar, no Google Drive, a pasta:
`Portifólio DS Vini/Challenge_ClickBus_2025/data/`
2. Executar, no Google Colab, os notebooks na exata ordem do pipeline Medallion:
1. `01_Camada_Bronze`
2. `02_Camada_Prata`
3. `03_Camada_Ouro_EDA`
4. `04_Camada_Ouro_Segmentacao`
5. `05_Camada_Ouro_Timing_e_Tudo`



Cada notebook lê apenas os arquivos `.parquet` gerados pela etapa anterior.

---

## Tecnologias

A stack foi consolidada para abranger desde a limpeza pesada de dados até o balanceamento robusto e explainable AI:

`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `statsmodels`, `scipy`, `holidays`, `pyarrow`, `xgboost`, `imbalanced-learn`, `shap`

*(O modelo preditivo de próximo trecho, em desenvolvimento, utilizará também `lightgbm`.)*

---

## Autor

**Vinicius de Sousa Macedo** Tecnólogo em Data Science (FIAP) · Bacharel em Economia

* LinkedIn: [https://linkedin.com/in/vsmacedo](https://linkedin.com/in/vsmacedo)
* GitHub: [https://github.com/vsmacedo-datafinance](https://github.com/vsmacedo-datafinance)

*Dados fornecidos pela ClickBus para fins acadêmicos e de portfólio profissional, anonimizados em total conformidade com a LGPD.*
