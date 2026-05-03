Contexto
A ClickBus é a maior plataforma de venda de passagens rodoviárias do Brasil, com mais de 62 milhões de bilhetes emitidos, presença em 4.000 destinos e parceria com mais de 200 viações. O transporte rodoviário é o modal dominante no país — são estimados 170 milhões de tickets/ano, num mercado de R$ 20 bilhões de GMV, dos quais 65–70% ainda ocorrem offline.
Este projeto foi desenvolvido como resposta ao Innovation Challenge 2025 da FIAP, com dados reais (anonimizados) fornecidos pela ClickBus. A refatoração v2 aplica rigor analítico e econômico sobre a solução original, reorganizando o pipeline na Arquitetura Medalhão (Bronze → Prata → Ouro) com três objetivos de negócio:
#DesafioPergunta de negócio1SegmentaçãoQuais são os perfis de viajante e como personalizar estratégias de Growth para cada um?2Previsão de RecompraEste cliente vai comprar nos próximos 30 dias? Qual a probabilidade?3Próximo TrechoQual rota esse cliente provavelmente vai comprar na próxima viagem?

Diferenciais do Projeto
Este projeto vai além da aplicação técnica de modelos. A formação em Economia orienta a interpretação dos dados:

🏛️ Variáveis exógenas vs. endógenas — COVID-19 tratado como choque exógeno estrutural, não como ruído
📐 Framework RFM estendido — com proxies econômicas: score_fidelidade (market share individual) e hhi_rotas (concentração de destinos)
💰 Threshold de negócio — ponto de corte do modelo binário calibrado pela relação entre CAC e ticket médio, não pelo padrão 0.5
🧑‍🤝‍🧑 Clusters nomeados com persona — não "Cluster 2", mas "Heavy User Volátil" com estratégia de Growth específica


Arquitetura — Pipeline Medalhão
CSV Bruto (1.7M linhas)
        │
        ▼
┌───────────────────────────────────┐
│  🥉 BRONZE                        │
│  Ingestão · Integridade · Schema  │
│  → clickbus_bronze.parquet        │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  🥈 PRATA                         │
│  Limpeza · Features determinísticas│
│  Split out-of-time (sem leakage)  │
│  → clickbus_treino.parquet        │
│  → clickbus_val.parquet           │
│  → clickbus_teste.parquet         │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  🥇 OURO — EDA                    │
│  Análise exploratória (só treino) │
│  Feature Engineering por cliente  │
│  RFM + Proxies econômicas         │
│  VIF · Correlação · Skewness      │
│  → df_cliente.parquet             │
│  → df_cluster.parquet             │
└───────────────────────────────────┘
        │
   ┌────┼────────────┐
   ▼    ▼            ▼
[M1]  [M2]         [M3]
K-Means XGBoost  LightGBM
Princípio central: todo cálculo estatístico (scores de fidelidade, popularidade de trechos, proporções de comportamento) é feito exclusivamente sobre df_treino e aplicado em validação e teste via .map(). Isso elimina por design a classe de erros de data leakage que comprometia a versão original.

Estrutura do Repositório
Challenge_ClickBus_FIAP_2025/
│
├── src/
│   └── utils.py                          # Funções modulares compartilhadas
│
├── notebooks/
│   ├── 01_Camada_Bronze.ipynb            # Ingestão e integridade
│   ├── 02_Camada_Prata.ipynb             # Limpeza, features e split temporal
│   ├── 03_Camada_Ouro_EDA.ipynb          # EDA orientada a negócio + df_cliente
│   ├── 04_Camada_Ouro_Segmentacao.ipynb  # Modelo 1 — K-Means (5 clusters)
│   ├── 05_Camada_Ouro_Classificacao.ipynb# Modelo 2 — XGBoost (recompra 30d)
│   └── 06_Camada_Ouro_Multiclasse.ipynb  # Modelo 3 — LightGBM (próximo trecho)
│
└── README.md

Notebooks — O que cada um faz
🥉 01 · Camada Bronze — Ingestão e Integridade
Extrai o CSV bruto (~1.7M linhas, 2013–2024), valida schema e salva em Parquet com checagem de integridade. A camada Bronze preserva o dado original sem transformação — é a fonte da verdade auditável do projeto.
Destaques:

Download do dataset via gdown direto do Google Drive
Checagem de período, nulls e shape antes de qualquer transformação
Saída em Parquet (10× mais eficiente que CSV para leitura subsequente)


🥈 02 · Camada Prata — Limpeza e Preparação
Recebe o Bronze e aplica as regras de negócio para produzir um dado confiável para análise.
Destaques:

Decodificação de hashes anonimizados (cidades, empresas, clientes) via mapear_hashes()
Remoção de registros inconsistentes com log estruturado de cada regra aplicada:

RegraCritérioGMV inválidogmv_success <= 0Origem = DestinoViagem sem deslocamento realDados de ida ausentesRegistro incompleto

Criação de features determinísticas (feriado, hora do dia, trecho, empresa) — aqui e não na Ouro, pois não dependem de estatísticas do treino
Split out-of-time respeitando a ordem temporal: treino → validação → teste, sem embaralhar
Variável periodo_covid como choque exógeno estrutural (pré / durante / pós pandemia)


🥇 03 · Camada Ouro — EDA e Feature Engineering
A EDA é conduzida exclusivamente sobre df_treino. Toda análise tem dupla camada: técnica + interpretação econômica.
Análise transacional:

Sazonalidade mensal e semanal do GMV
Concentração de rotas (regra de Pareto 80/20)
Market share de operadoras por faturamento e volume
Quebra estrutural da pandemia — série histórica de GMV com banda COVID marcada

Construção do df_cliente — Framework RFM estendido:
DimensãoVariáveisInterpretação econômicaMonetarygmv_total, gmv_total_logValor histórico gerado pelo clienteFrequencytotal_compras, total_compras_logIntensidade de uso da plataformaRecencyrecencia_dias, tempo_vida_diasEngajamento recente e longevidadeComportamenton_destinos_distintos, prop_feriado, prop_ida_voltaPerfil turístico e elasticidadeFidelidadescore_fidelidadeMarket share individual por cliente (GMV empresa favorita / GMV total)
Rigor estatístico:

Análise de skewness → transformação log1p nas variáveis assimétricas
Correlação Pearson + Spearman (linear e monotônica)
VIF (Variance Inflation Factor) para diagnóstico de multicolinearidade
Remoção de variáveis redundantes com justificativa econômica por par


🥇 04 · Segmentação — K-Means (5 Clusters)
Problema: identificar perfis de viajante distintos para personalizar estratégias de Growth.
Metodologia:

StandardScaler obrigatório — K-Means usa distância euclidiana, sensível à escala
K escolhido por três critérios combinados: Elbow (inércia) + Silhouette Score + Davies-Bouldin Index
Comparação de K = 4, 5 e 6 com tabela de perfis medianos

Os 5 perfis identificados:
ClusterNomeEstratégia de Growth0Recorrente ExplorerCross-sell de rotas — recomendação personalizada tem alta conversão1Heavy User VolátilPrograma de fidelidade — capturar share of wallet2Família / Grupo (Multi-ticket)Oferta de desconto por volume de tickets por compra3One-Shot de FeriadoCampanha sazonal — ativar 30–45 dias antes de feriados prolongados4One-Shot RegularReativação — cupom de desconto para segunda viagem
Visualizações:

Projeção PCA 2D (variância retida informada)
Radar chart do DNA econômico de cada cluster (índice relativo à média da base)
Box plots de coesão intra-cluster
Concentração de clientes vs GMV por perfil
Composição por período COVID — validação de robustez dos perfis


🥇 05 · Previsão de Recompra — XGBoost (em desenvolvimento)
Problema: classificação binária — o cliente vai comprar nos próximos 30 dias?
Metodologia:

Target construído a partir de df_val: verifica se cada cliente do treino aparece na janela de 30 dias posterior à data de corte
SMOTE aplicado apenas no conjunto de treino após o split
Threshold definido pela relação custo de campanha / ticket médio esperado (não pelo padrão 0.5)
SHAP values para interpretabilidade das features mais importantes


🥇 06 · Próximo Trecho — LightGBM Multiclasse (em desenvolvimento)
Problema: dado o histórico do cliente, qual dos top-10 trechos ele mais provavelmente vai comprar?
Metodologia:

Feature de estado: ultimo_destino_ida como contexto da última viagem
Foco nos 10 trechos mais frequentes (80% do volume)
Avaliação por F1-Score macro e Top-3 Accuracy


Tecnologias
CategoriaBibliotecasManipulação de dadospandas, numpyMachine Learningscikit-learn, xgboost, lightgbm, imbalanced-learnInterpretabilidadeshapVisualizaçãomatplotlib, seabornEstatísticastatsmodels, scipyCalendárioholidaysArmazenamentopyarrow (Parquet)AmbienteGoogle Colab + Google Drive

Como Executar
Os notebooks são projetados para rodar no Google Colab com dados armazenados no Google Drive.
1. Clone o repositório (feito automaticamente pelo notebook via !git clone):
bashgit clone https://github.com/vsmacedo-datafinance/Challenge_ClickBus_FIAP_2025.git
2. Execute os notebooks em ordem:
01_Bronze → 02_Prata → 03_Ouro_EDA → 04_Segmentacao → 05_Classificacao → 06_Multiclasse
Cada notebook lê a saída do anterior via Parquet — não é necessário rodar tudo de uma vez.
3. Estrutura de pastas esperada no Drive:
Portifólio DS Vini/Challenge_ClickBus_2025/data/
├── bronze/   → clickbus_bronze.parquet
├── prata/    → clickbus_treino.parquet · clickbus_val.parquet · clickbus_teste.parquet
└── ouro/     → df_cliente.parquet · df_cluster.parquet · df_cliente_clusterizado.parquet

Sobre o Projeto
Desenvolvido como projeto de portfólio durante o 2º ano do Tecnólogo em Data Science na FIAP, com base no Innovation Challenge 2025 realizado em parceria com a ClickBus.
A refatoração v2 foi conduzida com foco em rigor metodológico e interpretação econômica — unindo a formação em Economia com o aprendizado técnico em Data Science.
Autor: Vinicius de Sousa Macedo
LinkedIn: linkedin.com/in/vsmacedo
GitHub: github.com/vsmacedo-datafinance

Dados fornecidos pela ClickBus para fins acadêmicos — anonimizados conforme LGPD.
