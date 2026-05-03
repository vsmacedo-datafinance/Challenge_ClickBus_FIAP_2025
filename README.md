ClickBus — Decodificando o Comportamento do Viajante  
FIAP Innovation Challenge 2025 · Refatoração v2 · Portfólio de Data Science  

***

## Contexto

A ClickBus é a maior plataforma de venda de passagens rodoviárias do Brasil, atuando em um mercado estimado em 170 milhões de tickets/ano e R$ 20 bilhões em GMV, ainda majoritariamente offline. Este projeto usa dados reais anonimizados do Innovation Challenge FIAP 2025 para entender o comportamento de compra e apoiar decisões de Growth.

Foco desta versão: **arquitetura de dados + EDA + segmentação de clientes**.  
Os modelos de **recompra** e **próximo trecho** estão planejados, mas ainda em desenvolvimento.

***

## Versão refatorada (v2) vs. versão original

A solução original foi entregue no 1º ano da graduação, com a maior parte da lógica centralizada em um único notebook e risco de data leakage.

A refatoração v2, desenvolvida no 2º ano, reorganiza o projeto em:

- **Arquitetura Medalhão:** Bronze (ingestão); Prata (limpeza); Ouro (EDA e modelos).  
- **Código modular:** funções de regra de negócio em `src/utils.py`.  
- **Prevenção de leakage por design:** toda estatística (popularidade de trechos, scores de fidelidade, proporções de comportamento) é calculada apenas em `df_treino` e aplicada em validação/teste via `.map()`.

Os modelos continuam os mesmos (K-Means na segmentação), mas a disciplina de dados e a leitura econômica evoluíram.

***

## Diferenciais analíticos

- **Visão de Economia aplicada a dados**
  - `periodo_covid` (pré / durante / pós) tratado como choque exógeno estruturante.  
  - `score_fidelidade` como proxy de market share individual (GMV da empresa favorita / GMV total do cliente).  
  - Leitura de trade-offs de Growth: custo de campanha vs. valor esperado de cliente.

- **Framework RFM estendido em nível de cliente**
  - Monetary: `gmv_total`, `gmv_total_log`.  
  - Frequency: `total_compras`, `total_compras_log`.  
  - Recency: `recencia_dias`, `tempo_vida_dias`.  
  - Comportamento: `n_destinos_distintos`, `prop_feriado`, `prop_ida_volta`, `prop_fim_semana`.  
  - Fidelidade: `score_fidelidade`.

- **Rigor estatístico na EDA**
  - Transformação `log1p` em variáveis altamente assimétricas.  
  - Correlações Pearson e Spearman para relações lineares/monotônicas.  
  - VIF para diagnóstico de multicolinearidade e remoção de variáveis redundantes com justificativa.

***

## Arquitetura e notebooks (até Segmentação)

- **01_Camada_Bronze.ipynb — Ingestão e integridade**  
  - Lê o CSV bruto (~1,7M linhas, 2013–2024).  
  - Valida schema, nulos e intervalo de datas.  
  - Salva `clickbus_bronze.parquet` como fonte imutável.

- **02_Camada_Prata.ipynb — Limpeza e preparação**  
  - Decodifica hashes de cidades, empresas e clientes.  
  - Remove registros inconsistentes (GMV ≤ 0, origem = destino, dados de ida ausentes) com log de impacto.  
  - Cria features determinísticas (`e_feriado`, `compra_ate_5_dias_feriado`, `trecho_ida`, `empresa_ida`).  
  - Realiza split temporal out-of-time: treino; validação; teste (sem embaralhar).  
  - Define `periodo_covid` (pré / durante / pós) para capturar o choque da pandemia.

- **03_Camada_Ouro_EDA.ipynb — EDA + df_cliente**  
  - EDA apenas em `df_treino` para evitar leakage.  
  - Análises em nível transação: sazonalidade, concentração de rotas, participação de empresas, série de GMV com banda COVID.  
  - Construção de `df_cliente` agregando RFM + comportamento + fidelidade.  
  - Saídas:
    - `df_cliente.parquet`: visão completa por cliente.  
    - `df_cluster.parquet`: subset numérico preparado para K-Means.

- **04_Camada_Ouro_Segmentacao.ipynb — K-Means (5 clusters)**  
  - Carrega `df_cluster.parquet` (id_cliente + 11 features numéricas).  
  - Aplica `StandardScaler` (K-Means é sensível à escala).  
  - Testa k de 3 a 10 com:
    - Método do cotovelo (inércia).  
    - Silhouette Score.  
  - Compara perfis agregados para k = 4, 5, 6 e fixa **k = 5** como compromisso entre qualidade de separação e clareza de personas para Growth.  
  - Salva `df_cliente_clusterizado.parquet` (id_cliente + features + cluster + nome_cluster).

***

## Segmentação — K-Means (k = 5)

Modelo: K-Means sobre as features numéricas de cliente (logs de valor/frequência/recência/diversidade, proporções de feriado/fim de semana/ida-volta, ticket médio, score de fidelidade), após padronização.

**Personas identificadas**

| Cluster | Nome                     | Características principais | Uso de negócio |
|--------:|--------------------------|----------------------------|----------------|
| 0       | Recorrente Explorer      | Valor e frequência acima da média, mais destinos distintos | Cross-sell de rotas, recomendações personalizadas |
| 1       | Heavy User Volátil       | Maior GMV e frequência da base, baixa fidelidade a empresa | Programa de fidelidade, captura de share of wallet |
| 2       | Famílias / Grupo         | Ticket médio alto, forte uso de ida-volta, multi-tickets | Combos, descontos progressivos, comunicação para grupos |
| 3       | One-shot de Feriado      | Compra única concentrada em feriados | Campanhas sazonais 30–45 dias antes de feriados prolongados |
| 4       | One-shot Regular         | Compra única fora de feriado, fidelidade ligeiramente maior | Campanhas de reativação e incentivo à segunda viagem |

**Validações e visualizações**

- Projeção PCA 2D (~50% da variância explicada):  
  - Heavy Users e Famílias aparecem bem separados em relação aos demais.  
  - One-shot Regular e One-shot de Feriado se aproximam no espaço PCA, o que é coerente: a principal diferença entre eles é mais comportamental (calendário) do que de volume.  
- “DNA dos clusters”: tabela de índices (cluster / média global) para valor, frequência, ticket médio, fidelidade, foco em feriados e ida-volta.  
- Estabilidade dos segmentos por `periodo_covid_max` (pré-covid vs durante-covid):  
  - Todos os perfis aparecem em ambos os períodos.  
  - Heavy Users ganham peso relativo durante a pandemia, enquanto One-shot Regular perde participação, sugerindo concentração de receita em clientes recorrentes sob choque adverso.

***

## Como executar (até Segmentação)

1. Preparar, no Google Drive, a pasta:

   `Portifólio DS Vini/Challenge_ClickBus_2025/data/`

2. Executar, no Google Colab, os notebooks na ordem:

   01_Camada_Bronze - 02_Camada_Prata - 03_Camada_Ouro_EDA - 04_Camada_Ouro_Segmentacao  

Cada notebook lê apenas os arquivos Parquet gerados pela etapa anterior.

***

## Tecnologias

`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`,  
`statsmodels`, `scipy`, `holidays`, `pyarrow`  

*(Modelos de classificação previstos utilizarão também `xgboost`, `lightgbm`, `imbalanced-learn`, `shap`.)*

***

## Autor

**Vinicius de Sousa Macedo**  
Tecnólogo em Data Science (FIAP) · Bacharel em Economia  

- LinkedIn: https://linkedin.com/in/vsmacedo  
- GitHub: https://github.com/vsmacedo-datafinance  

Dados fornecidos pela ClickBus para fins acadêmicos, anonimizados conforme LGPD.
