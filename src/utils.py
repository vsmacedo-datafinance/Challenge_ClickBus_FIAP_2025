import pandas as pd
import numpy as np
import holidays
import json
import time

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
)

# Constantes globais
INICIO_COVID = pd.Timestamp('2020-03-11')
FIM_COVID    = pd.Timestamp('2023-05-05')

# Funções

def mapear_hashes(df, colunas, prefixo, valor_nulo='0'):
    """
    Substitui hashes por labels legíveis (ex: Cidade_0).
    Retorna o dataframe com as novas colunas e o dicionário de mapeamento.
    """
    hashes_validos = np.concatenate([
        df[col][df[col] != valor_nulo].unique() for col in colunas
    ])
    
    mapeamento = {h: i for i, h in enumerate(pd.Series(hashes_validos).unique())}

    def _legenda(hash_val):
        if pd.isna(hash_val) or hash_val == valor_nulo:
            return valor_nulo
        return f"{prefixo}_{mapeamento.get(hash_val, 'NA')}"

    df_result = df.copy()
    for col in colunas:
        df_result[f"{col}_legend"] = df_result[col].map(_legenda)

    return df_result, mapeamento


def tratar_dados_clickbus(df):
    """
    Aplica as regras de negócio: mapeamento de cidades, empresas e flag de Covid.
    """
    # Cidades
    cols_cidade = [
        'place_origin_departure', 'place_destination_departure',
        'place_origin_return', 'place_destination_return'
    ]
    df, map_cidades = mapear_hashes(df, cols_cidade, 'Cidade')

    # Empresas
    cols_empresa = ['fk_departure_ota_bus_company', 'fk_return_ota_bus_company']
    df, map_empresas = mapear_hashes(df, cols_empresa, 'Empresa')

    # Clientes (cat.codes garante um ID único sequencial)
    df['fk_contact_legend'] = 'Cliente_' + df['fk_contact'].astype('category').cat.codes.astype(str)

    #  Período COVID
    col_data = 'date_purchase' if 'date_purchase' in df.columns else 'data_compra'
    
    def classificar_covid(data):
        if data < INICIO_COVID:
            return 'pre-covid'
        elif data <= FIM_COVID:
            return 'durante-covid'
        else:
            return 'pos-covid'

    df['periodo_covid'] = df[col_data].apply(classificar_covid)

    dicionarios = {'cidades': map_cidades, 'empresas': map_empresas}
    
    return df, dicionarios


def split_temporal(df, test_size, val_size, col_data='data_compra'):
    """
    Divide os dados em treino, validação e teste respeitando a linha do tempo.
    """
    # Ordena por data para evitar vazamento de dados (Data Leakage)
    df = df.sort_values(by=col_data).reset_index(drop=True)

    df_test  = df.tail(test_size).copy()
    df_temp  = df.iloc[:-test_size]
    df_val   = df_temp.tail(val_size).copy()
    df_train = df_temp.iloc[:-val_size].copy()

    df_train['split'] = 'train'
    df_val['split'] = 'val'
    df_test['split'] = 'test'

    print(f"Treino:    {df_train.shape[0]} linhas")
    print(f"Validação: {df_val.shape[0]} linhas")
    print(f"Teste:     {df_test.shape[0]} linhas")

    return df_train, df_val, df_test


def salvar_mapeamentos(mapeamentos, caminho):
    """Salva os dicionários em um arquivo JSON."""
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(mapeamentos, f, ensure_ascii=False, indent=4)
    print(f"Mapeamentos salvos em: {caminho}")


def carregar_mapeamentos(caminho):
    """Lê os dicionários de um arquivo JSON."""
    with open(caminho, 'r', encoding='utf-8') as f:
        return json.load(f)

def enriquecer_dados_temporais(df: pd.DataFrame, col_data: str = 'data_compra', dias_antecedencia: int = 5) -> pd.DataFrame:
    df_temp = df.copy()

    df_temp['ano'] = df_temp[col_data].dt.year.astype('int16')
    df_temp['mes'] = df_temp[col_data].dt.month.astype('int8')
    df_temp['dia'] = df_temp[col_data].dt.day.astype('int8')
    df_temp['dia_semana'] = df_temp[col_data].dt.dayofweek.astype('int8') # 0=Segunda, 6=Domingo
    df_temp['fim_de_semana'] = df_temp['dia_semana'].isin([5, 6]).astype('int8')

    ano_min = df_temp['ano'].min()
    ano_max = df_temp['ano'].max()
    feriados_br = holidays.Brazil(years=range(ano_min, ano_max + 1))
    feriados_set = set(feriados_br.keys())
    
    df_temp['e_feriado'] = df_temp[col_data].dt.date.isin(feriados_set).astype('int8')
    datas_janela = set()
    for data_feriado in feriados_set:
        datas_range = pd.date_range(end=data_feriado, periods=dias_antecedencia + 1).date
        datas_janela.update(datas_range)
        
    nome_coluna_delta = f'compra_ate_{dias_antecedencia}_dias_feriado'
    df_temp[nome_coluna_delta] = df_temp[col_data].dt.date.isin(datas_janela).astype('int8')

    return df_temp

def analisar_feriados_projeto(df_treino, data_corte, anos_historico=range(2013, 2025)):

    feriados_br_dict = holidays.Brazil(years=anos_historico)

    compras_feriados = df_treino[df_treino['e_feriado'] == 1].copy()

    feriados_stats = (
        compras_feriados
        .groupby(compras_feriados['data_compra'].dt.date)
        .agg(
            gmv_total    = ('gmv_success', 'sum'),
            num_compras  = ('gmv_success', 'count'),
            tickets_total= ('quantidade_tickets', 'sum')
        )
        .reset_index()
    )

    feriados_stats['nome_feriado'] = feriados_stats['data_compra'].map(feriados_br_dict)

    # Remove datas que não mapearam para nenhum feriado (edge case de feriados estaduais)
    feriados_stats = feriados_stats.dropna(subset=['nome_feriado'])

    top_feriados = (
        feriados_stats
        .groupby('nome_feriado')
        .agg(
            gmv_medio_anual      = ('gmv_total', 'mean'),
            compras_medias_anual = ('num_compras', 'mean'),
            tickets_medios_anual = ('tickets_total', 'mean'),
            anos_ocorreu         = ('data_compra', 'count')
        )
        .round(2)
        .sort_values('gmv_medio_anual', ascending=False)
    )

    # sorted() é O(n log n) sobre lista pequena de feriados — ok
    feriados_lista = sorted(feriados_br_dict.keys())
    data_ref = pd.to_datetime(data_corte).date()

    proximo_feriado  = next((f for f in feriados_lista if f > data_ref), None)
    dias_ate_feriado = (proximo_feriado - data_ref).days if proximo_feriado else 999

    return top_feriados, proximo_feriado, dias_ate_feriado


# MODELO — Timing é Tudo

def enriquecer_features_modelo2(
    df_cliente: pd.DataFrame,
    df_treino: pd.DataFrame,
    col_id: str = 'id_cliente',
) -> pd.DataFrame:
    # 1. sazonalidade_score
    MESES_ALTA = [1, 7, 12]
    df_treino = df_treino.copy()
    df_treino['compra_mes_alta'] = df_treino['mes'].isin(MESES_ALTA).astype(int)
 
    sazon = (
        df_treino
        .groupby(col_id)
        .agg(compras_alta=('compra_mes_alta', 'sum'), total=('gmv_success', 'count'))
        .assign(sazonalidade_score=lambda x: x['compras_alta'] / x['total'])
        .reset_index()[[col_id, 'sazonalidade_score']]
    )
 
    # 2. prop_ferias
    MESES_FERIAS = [1, 7, 11, 12]
    df_treino['mes_ferias'] = df_treino['mes'].isin(MESES_FERIAS).astype(int)
 
    ferias = (
        df_treino
        .groupby(col_id)
        .agg(compras_ferias=('mes_ferias', 'sum'), total_f=('gmv_success', 'count'))
        .assign(prop_ferias=lambda x: x['compras_ferias'] / x['total_f'])
        .reset_index()[[col_id, 'prop_ferias']]
    )
 
    # 3. mes_ultima_compra
    mes_ult = (
        df_treino
        .groupby(col_id)['data_compra']
        .max()
        .dt.month
        .reset_index()
        .rename(columns={'data_compra': 'mes_ultima_compra'})
    )
 
    # 4. intervalo_medio_dias
    # .diff() dentro de cada grupo calcula dias entre compras consecutivas.
    # A média desses intervalos é o ritmo de compra do cliente.
    # Clientes com 1 compra têm NaN (sem diferença calculável) → preenchido com 9999.
    intervalo = (
        df_treino
        .sort_values([col_id, 'data_compra'])
        .assign(dias_prev=lambda x:
            x.groupby(col_id)['data_compra'].diff().dt.days
        )
        .groupby(col_id)['dias_prev']
        .mean()
        .fillna(9999)
        .reset_index()
        .rename(columns={'dias_prev': 'intervalo_medio_dias'})
    )
 
    # 5. dias_ate_proximo_feriado — constante para todos os clientes
    _, _, dias_ate_feriado = analisar_feriados_projeto(
        df_treino, data_corte=df_treino['data_compra'].max()
    )
 
    # Merge sequencial — left join preserva todos os clientes
    df_out = (
        df_cliente
        .merge(sazon,    on=col_id, how='left')
        .merge(ferias,   on=col_id, how='left')
        .merge(mes_ult,  on=col_id, how='left')
        .merge(intervalo, on=col_id, how='left')
    )
 
    df_out['dias_ate_proximo_feriado'] = dias_ate_feriado
 
    df_out['intervalo_medio_dias']  = df_out['intervalo_medio_dias'].fillna(9999)
    df_out['sazonalidade_score']    = df_out['sazonalidade_score'].fillna(0)
    df_out['prop_ferias']           = df_out['prop_ferias'].fillna(0)
 
    print(f"df_cliente shape após enriquecimento: {df_out.shape}")
    print(f"Novas features: sazonalidade_score, prop_ferias, mes_ultima_compra, "
          f"intervalo_medio_dias, dias_ate_proximo_feriado")
 
    return df_out

def construir_target_30d(
    df_cliente_clust: pd.DataFrame,
    df_val: pd.DataFrame,
    data_ref: pd.Timestamp,
    janela_dias: int = 30,
    col_data: str = 'data_compra',
    col_id: str = 'id_cliente',
) -> pd.DataFrame:
    data_limite = data_ref + pd.Timedelta(days=janela_dias)

    compradores = set(
        df_val
        .loc[df_val[col_data] <= data_limite, col_id]
        .unique()
    )

    df_out = df_cliente_clust.copy()
    df_out['target'] = df_out[col_id].isin(compradores).astype(int)

    n_total = len(df_out)
    n_pos   = df_out['target'].sum()

    print(f"data_ref    : {data_ref.date()}")
    print(f"data_limite : {data_limite.date()} (+{janela_dias}d)")
    print(f"Compradores na janela : {len(compradores):,}")
    print(f"Target = 1  : {n_pos:,} ({n_pos/n_total:.1%})")
    print(f"Target = 0  : {n_total-n_pos:,} ({(n_total-n_pos)/n_total:.1%})")
    print(f"Ratio       : 1 : {int((n_total-n_pos)/n_pos)}")

    return df_out

def preparar_features(
    df_modelo: pd.DataFrame,
    cols_excluir: list,
    col_target: str = 'target',
    col_cluster: str = 'cluster',
):
    df_enc = pd.get_dummies(
        df_modelo,
        columns=[col_cluster],
        prefix=col_cluster,
        drop_first=False,
    )

    cols_remover = [c for c in cols_excluir + [col_target] if c in df_enc.columns]
    features = [c for c in df_enc.columns if c not in cols_remover]

    X = df_enc[features]
    y = df_enc[col_target]

    assert X.isnull().sum().sum() == 0, "Nulos encontrados nas features!"

    print(f"Features : {X.shape[1]}")
    print(f"Amostras : {X.shape[0]:,}")
    print(f"Positivos: {y.sum():,} ({y.mean():.1%})")

    return X, y


def avaliar_estrategias(
    experimentos: dict,
    X_train, y_train,
    X_eval, y_eval,
    param_grid: dict,
    cv,
    random_state: int = 42,
    n_iter: int = 10,
) -> dict:
    resultados = {}

    for nome, pipe in experimentos.items():
        print(f"Treinando {nome}...", end=' ')
        inicio = time.time()

        search = RandomizedSearchCV(
            pipe, param_grid,
            n_iter=n_iter,
            scoring='average_precision',
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        y_prob = search.best_estimator_.predict_proba(X_eval)[:, 1]
        y_pred = search.best_estimator_.predict(X_eval)

        resultados[nome] = {
            'modelo'   : search.best_estimator_,
            'y_prob'   : y_prob,
            'y_pred'   : y_pred,
            'auc_roc'  : roc_auc_score(y_eval, y_prob),
            'auc_pr'   : average_precision_score(y_eval, y_prob),
            'f1'       : f1_score(y_eval, y_pred),
            'recall'   : recall_score(y_eval, y_pred),
            'precision': precision_score(y_eval, y_pred, zero_division=0),
        }

        print(f"{time.time()-inicio:.0f}s | AUC-PR: {resultados[nome]['auc_pr']:.4f}")

    return resultados