ClickBus — Decodificando o Comportamento do Viajante  
FIAP Innovation Challenge 2025 · Refatoração v2 · Portfólio de Data Science  

***

## Visão geral

A ClickBus é a maior plataforma de passagens rodoviárias do Brasil, em um mercado estimado em 170 milhões de tickets/ano e R$ 20 bilhões em GMV, ainda majoritariamente offline. Este projeto refatora a solução do Innovation Challenge 2025 usando Arquitetura Medalhão (Bronze → Prata → Ouro) e foco em rigor estatístico e leitura econômica, até a etapa de segmentação de clientes.

**Desafios cobertos nesta versão**

1. Segmentação: identificar perfis de viajante e direcionar estratégias de Growth.  

(Os desafios de Recompra e Próximo Trecho serão adicionados em versões futuras.)

***

## Diferenciais

- Uso explícito de conceitos de Economia:
  - COVID-19 tratado como choque exógeno (pré / durante / pós), não como ruído.
  - Framework RFM estendido com proxies econômicas:
    - score_fidelidade (market share individual do cliente).
    - hhi_rotas (concentração de destinos).
- Arquitetura de dados clara:
  - Bronze: ingestão do CSV bruto, validação de integridade, persistência em Parquet.
  - Prata: limpeza, regras de negócio, features determinísticas e split temporal sem leakage.
  - Ouro (até aqui): EDA apenas em treino, feature engineering em nível de cliente e segmentação com K-Means.

***

## Arquitetura e Notebooks (até Segmentação)

- 01_Camada_Bronze.ipynb  
  - Lê o CSV bruto (~1,7M linhas, 2013–2024), valida schema, nulls e período.
  - Salva `clickbus_bronze.parquet` como base imutável.

- 02_Camada_Prata.ipynb  
  - Decodifica hashes anonimizados (cidades, empresas, clientes).  
  - Aplica regras de limpeza (GMV inválido, origem=destino, registros incompletos).  
  - Cria features determinísticas (feriado, hora do dia, trecho, empresa).  
  - Realiza split temporal out-of-time: treino → validação → teste.  
  - Inclui variável `periodo_covid` (pré / durante / pós).

- 03_Camada_Ouro_EDA.ipynb  
  - EDA apenas em `df_treino` para evitar data leakage.  
  - Construção de `df_cliente` (RFM estendido):  
    - Monetary: `gmv_total`, `gmv_total_log`.  
    - Frequency: `total_compras`, `total_compras_log`.  
    - Recency: `recencia_dias`, `tempo_vida_dias`.  
    - Comportamento: `n_destinos_distintos`, `prop_feriado`, `prop_ida_volta`.  
    - Fidelidade: `score_fidelidade`.  
  - Transformações log1p em variáveis altamente assimétricas.  
  - Análises de correlação (Pearson/Spearman) e VIF para reduzir multicolinearidade.  
  - Saídas:
    - `df_cliente.parquet`: visão completa por cliente.
    - `df_cluster.parquet`: subconjunto numérico preparado para clusterização.

- 04_Camada_Ouro_Segmentacao.ipynb  
  - Carrega `df_cluster.parquet` (id_cliente + 11 features numéricas).  
  - Padroniza as features com StandardScaler (K-Means usa distância euclidiana).  
  - Testa k de 3 a 10 com:
    - Método do cotovelo (inércia).
    - Silhouette Score.
  - Compara perfis agregados para k = 4, 5 e 6 e escolhe k = 5 pela combinação:
    - boa qualidade de separação pelas métricas;
    - personas mais interpretáveis para Growth.

***

## Segmentação — Perfis de Cliente (k = 5)

Modelo: K-Means com StandardScaler aplicado sobre:

- `gmv_total_log`, `total_compras_log`, `recencia_dias_log`, `tempo_vida_dias_log`,  
- `n_destinos_distintos_log`, `pop_trecho_media_log`,  
- `score_fidelidade`, `prop_fim_semana`, `prop_feriado`,  
- `tickets_medio`, `prop_ida_volta`.

Perfis identificados (resumo econômico):

- Cluster 0 – Recorrente Explorer  
  - Valor e frequência acima da média (≈1,1–1,3x).  
  - Mais destinos e relacionamento mais longo.  
  - Bom alvo para cross-sell de rotas.

- Cluster 1 – Heavy Users Voláteis  
  - Maior GMV e maior frequência da base (≈1,3x e ≈2,1x).  
  - Fidelidade abaixo da média.  
  - Prioridade para ações de fidelização e aumento de share of wallet.

- Cluster 2 – Famílias / Grupo (Multi-ticket)  
  - Ticket médio alto (≈2x) e forte uso de ida-e-volta.  
  - Foco em compras em grupo.  
  - Indicado para combos, descontos progressivos e ofertas familiares.

- Cluster 3 – One-shot de Feriado  
  - Baixa frequência, forte concentração em feriados (≈5,5x a média).  
  - Bom alvo para campanhas sazonais antes de feriados prolongados.

- Cluster 4 – One-shot Regular  
  - Baixa frequência e valor abaixo da média.  
  - Fidelidade ligeiramente acima da média.  
  - Público para campanhas de reativação de segunda viagem.

Principais visualizações:

- PCA 2D com variância retida informada (~50%), mostrando boa separação entre Heavy Users, Famílias e Recorrentes; One-shot Regular e One-shot de Feriado aparecem próximos, o que é coerente (diferença mais comportamental que de volume).  
- “DNA dos clusters” (índice de cada feature vs média global) usado para definir as personas.  
- Estabilidade dos segmentos por período COVID (`pre-covid` vs `durante-covid`), mostrando que todos os perfis existem em ambos os períodos, com aumento relativo de Heavy Users durante a pandemia.

***

## Como executar (até Segmentação)

1. Montar no Google Drive a pasta:

   `Portifólio DS Vini/Challenge_ClickBus_2025/data/`

2. Rodar os notebooks, nesta ordem, no Google Colab:

   - 01_Camada_Bronze.ipynb  
   - 02_Camada_Prata.ipynb  
   - 03_Camada_Ouro_EDA.ipynb  
   - 04_Camada_Ouro_Segmentacao.ipynb  

Cada camada lê os Parquets gerados pela anterior; não é necessário executar tudo de uma vez.

***

## Sobre o projeto

Projeto de portfólio desenvolvido no 2º ano do Tecnólogo em Data Science na FIAP, com dados reais anonimizados fornecidos pela ClickBus para fins acadêmicos. Esta versão foca em rigor metodológico, prevenção de leakage e segmentação de clientes com interpretação econômica.
