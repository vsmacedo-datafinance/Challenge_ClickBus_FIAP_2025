import pandas as pd
import numpy as np
import holidays
import json

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