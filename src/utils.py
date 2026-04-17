import pandas as pd
import numpy as np
import json

# Constantes COVID
INICIO_COVID = pd.Timestamp('2020-03-11')
FIM_COVID = pd.Timestamp('2023-05-05')


# Funções 

def mapear_hashes(df, colunas, prefixo, valor_nulo='0'):
    hashes_validos = np.concatenate([
        df[col][df[col] != valor_nulo].unique() for col in colunas
    ])
    mapeamento = {h: i for i, h in enumerate(pd.Series(hashes_validos).unique())}

    def _legenda(hash_val):
        if hash_val == valor_nulo: return valor_nulo
        return f'{prefixo} {mapeamento.get(hash_val, "NA")}'

    df = df.copy()
    for col in colunas:
        df[col + '_legend'] = df[col].map(_legenda)
    
    return df, mapeamento


def tratar_dados_clickbus(df):

    cols_cidade = ['place_origin_departure', 'place_destination_departure', 
                   'place_origin_return', 'place_destination_return']
    df, map_cidades = mapear_hashes(df, cols_cidade, 'Cidade', valor_nulo='0')

    cols_empresa = ['fk_departure_ota_bus_company', 'fk_return_ota_bus_company']
    df, map_empresas = mapear_hashes(df, cols_empresa, 'Empresa', valor_nulo='1')

    if 'data_compra' in df.columns:
        df['periodo_covid'] = df['data_compra'].apply(lambda x: 'pre-covid' if x < INICIO_COVID else ('durante-covid' if x <= FIM_COVID else 'pos-covid'))

    return df, {'cidades': map_cidades, 'empresas': map_empresas}