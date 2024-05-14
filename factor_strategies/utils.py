import logging
import os

import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')


def feriados_anbima():
    with open(os.path.join(DATA_PATH, "anbima-feriados.txt"), 'r') as f:
        dates_txt = f.read().split('\n')
        dates = [pd.to_datetime(dt, format="%d/%m/%Y") for dt in dates_txt]
    return dates


def get_market_data():
    logging.info('Baixando o histórico de preços...')
    df = pd.read_parquet(
        "https://persevera.s3.sa-east-1.amazonaws.com/factor_zoo.parquet",
        columns=['date', 'code', 'price_close', 'volume_traded']
    )
    return df


def get_indices(indices):
    df = pd.read_parquet(r"\\xpdocs\research\equities\Quant\_Cross Data\economatica-indices.parquet")
    df = df[np.isin(df['code'], indices)]
    df = df.pivot_table(index='date', columns='code', values='close_price')
    return df


def get_factor_zoo(self, shrink=True):
    logging.info('Baixando o factor zoo...')

    if shrink:
        logging.info('Reduzindo a dimensão do factor zoo...')
        cols_model = ['code', 'date'] + [item for sublist in list(self.factor_list.values()) for
                                                                 item in sublist]
        df = pd.read_parquet("https://persevera.s3.sa-east-1.amazonaws.com/factor_zoo.parquet", columns=cols_model)

        logging.info(f'Colunas do factor zoo: {list(df.columns)}')
    else:
        df = pd.read_parquet("https://persevera.s3.sa-east-1.amazonaws.com/factor_zoo.parquet")

    return df


def prepare_factor_zoo(self):
    logging.info('Preparando o Factor Zoo...')

    # Importa as bases
    members = get_investment_universe(self)
    sector_comparison = get_sectors(self)

    # Importa o zoo
    logging.info('Combinando todos os DataFrames...')
    factors = self.factor_list.copy()
    factors['basic'] = {'basic': ['code', 'date', '21d_median_dollar_volume_traded']}

    cols = [inner_value for key, value in factors.items() for inner_key, inner_value in value.items()]
    cols = sorted([item for sublist in cols for item in sublist])
    # zoo = pd.read_parquet(os.path.join(DATA_PATH, "factor_zoo.parquet"),
    #                       columns=cols)
    zoo = pd.read_parquet("https://persevera.s3.sa-east-1.amazonaws.com/factor_zoo.parquet",
                          columns=cols)

    # Une todas as bases
    df = (
        pd.merge(
            left=members,
            right=sector_comparison,
            left_on='code',
            right_index=True,
            how='left'
        )
        .merge(
            right=self.factor_relevance,
            left_on='sector_specifics',
            right_index=True,
            how='left'
        )
        .merge(
            right=zoo,
            on=['date', 'code'],
            how='left'
        )
    )

    logging.info('Tratando as datas de rebalanceamento...')

    # Filtra somente os dados existentes da data do rebalanceamento
    df = df[np.isin(df['date'], self.rebalance_interval['most_recent'])]

    # Substitui a data para a data do rebalanceamento
    de_para = pd.merge(
        left=self.rebalance_interval['most_recent'],
        right=self.rebalance_interval['rebalance'],
        left_index=True,
        right_index=True
    ).set_index('most_recent').to_dict('index')

    df['date'] = df['date'].map(de_para).apply(lambda x: x['rebalance'])
    df.reset_index(drop=True, inplace=True)

    if list(self.investment_universe)[0] == 'size_segment':
        col = df.pop("size_segment")
        df.insert(2, col.name, col)
    return df


def get_sectors(self):
    logging.info('Baixando a classificação setorial...')
    df = pd.read_excel(os.path.join(DATA_PATH, "cadastro-base.xlsx"), sheet_name='equities')
    df = df[['code', self.sector_comparison, self.sector_specifics]]
    df.columns = ['code', 'sector_comparison', 'sector_specifics']
    df.set_index('code', inplace=True)
    return df


def get_factor_relevance(self):
    logging.info('Identificando a relevância de cada subfator...')
    df = pd.read_excel(
        os.path.join(os.path.dirname(__file__), "factor_relevance.xlsx"),
        sheet_name=self.sector_specifics,
        header=1,
        index_col=4,
    )
    df.drop(df.filter(regex='Unnamed').columns, axis=1, inplace=True)
    df.drop(columns=['Factor', 'Definition', 'Preferred'], inplace=True)
    # df.fillna(0, inplace=True)
    df = df[df.index.notnull()].T

    # Identifica os fatores utilizados
    factors = self.factor_list.copy()
    cols = [inner_value for key, value in factors.items() for inner_key, inner_value in value.items()]
    cols = sorted([item for sublist in cols for item in sublist])

    df = df[cols]

    # Renomeia as colunas
    new_cols = [f'relevance_{col}' for col in cols]
    df.columns = new_cols
    return df


def get_eligible_stocks(liquidity_thresh, liquidity_lookback):
    df = pd.read_parquet(
        "https://persevera.s3.sa-east-1.amazonaws.com/factor_zoo.parquet",
        columns=['price_close', 'volume_traded', '21d_median_dollar_volume_traded', f'{liquidity_lookback}d_median_dollar_volume_traded', 'market_cap']
    )

    # Calculate minimum required ADTV based on liquidity threshold
    turnover_threshold = df.groupby('date')['volume_traded'].quantile(q=liquidity_thresh).rolling(window=liquidity_lookback).mean().dropna()
    turnover_threshold.name = 'liquidity_threshold'

    df = pd.merge(
        left=df,
        right=turnover_threshold,
        left_index=True,
        right_index=True,
        how='left'
    )

    # Drop duplicates
    df.reset_index(inplace=True)
    df['radical'] = df['code'].str[:4]
    df.sort_values(by=['date', 'radical', '21d_median_dollar_volume_traded'], inplace=True)
    df.dropna(subset='21d_median_dollar_volume_traded', inplace=True)
    df.drop_duplicates(['radical', 'date'], keep='last', inplace=True)

    # Check eligibility
    df['eligible'] = df[f'{liquidity_lookback}d_median_dollar_volume_traded'] > df['liquidity_threshold']
    df = df.query('eligible == True')

    df['market_cap'] = df.groupby('code')['market_cap'].ffill(limit=5)
    df['21d_average_market_cap'] = df.groupby('code')['market_cap'].transform(lambda s: s.rolling(21, min_periods=1).mean())

    comparison_feature = '21d_average_market_cap'
    df = df.sort_values(['date', comparison_feature]).dropna(subset=[comparison_feature])
    df.set_index(['date', 'code'], inplace=True)

    # Calculate market cap ranks and size segments
    df['market_cap_rank'] = df.groupby('date')[comparison_feature].rank(pct=True)
    df['size_segment'] = pd.cut(df['market_cap_rank'], bins=[0, 1/3, 2/3, 1], labels=['Small', 'Mid', 'Large'])
    df = df['size_segment'].swaplevel().reset_index()

    return df
    
    
def get_investment_universe(self):
    logging.info('Calculando os membros do universo investível...')
    data = self.investment_universe
    
    df = get_eligible_stocks(data['liquidity_thresh'], data['liquidity_lookback'])
    segment_selection = ['Large', 'Mid', 'Small'] if data['size_segment'] == 'ALL' else [data['size_segment']]
    df = df[np.isin(df['size_segment'], segment_selection)]
    df = df.sort_values(['date', 'size_segment', 'code']).reset_index(drop=True)
    return df


def get_zscores(self, df, cols, cap=3):
    for col in cols:
        if 'high' in col:
            df[f'{col}_absolute_zscore'] = (df[col] - df[col].mean()) / df[col].std()
            df[f'{col}_relative_zscore'] = df.groupby('sector_comparison')[col].transform(lambda x: (x - x.mean()) / x.std())
        elif 'low' in col:
            df[f'{col}_absolute_zscore'] = -(df[col] - df[col].mean()) / df[col].std()
            df[f'{col}_relative_zscore'] = df.groupby('sector_comparison')[col].transform(lambda x: -(x - x.mean()) / x.std())

        # Limita o zscore
        df[f'{col}_absolute_zscore'] = df[f'{col}_absolute_zscore'].clip(lower=-cap, upper=cap)
        df[f'{col}_relative_zscore'] = df[f'{col}_relative_zscore'].clip(lower=-cap, upper=cap)

    return df


def get_percentiles(df_measures, cols, cap=3):
    df = df_measures.copy()
    for col in cols:
        if 'high' in col:
            df[f'{col}_absolute_rank'] = df[col].rank(pct=True) * 100
            df[f'{col}_relative_rank'] = df.groupby('sector_comparison')[col].rank(pct=True) * 100
        elif 'low' in col:
            df[f'{col}_absolute_rank'] = df[col].rank(pct=True, ascending=False) * 100
            df[f'{col}_relative_rank'] = df.groupby('sector_comparison')[col].rank(pct=True, ascending=False) * 100

    return df


def is_business_day(date, holiday):
    return bool(len(pd.bdate_range(start=date, end=date, freq='C', holidays=holiday)))


def get_rebalance_interval(self, business_day=0):
    logging.info('Identificando as datas de rebalanceamento...')
    holidays = feriados_anbima()
    dates = pd.bdate_range(start=self.start, end='2025-01-01', freq='C', holidays=holidays)
    df = pd.DataFrame(dates)
    df["month"] = df[0].dt.month
    df["year"] = df[0].dt.year

    try:
        df = [
            # Rebalance date
            pd.DataFrame(
                df.groupby(by=['year', 'month']).apply(lambda x: x.iloc[[business_day]] if len(x) > 1 else x)[0]),

            # Most recent
            pd.DataFrame(
                df.groupby(by=['year', 'month']).apply(lambda x: x.iloc[[business_day - 1]] if len(x) > 1 else x)[0]),
        ]
    except IndexError:
        s_month = str(df[0].max().month).zfill(2)
        s_year = str(df[0].max().year)
        df = df[df[0] < f'{s_year}-{s_month}']
        df = [
            # Rebalance date
            pd.DataFrame(
                df.groupby(by=['year', 'month']).apply(lambda x: x.iloc[[business_day]] if len(x) > 1 else x)[0]),

            # Most recent
            pd.DataFrame(
                df.groupby(by=['year', 'month']).apply(lambda x: x.iloc[[business_day - 1]] if len(x) > 1 else x)[0]),
        ]

    df = pd.merge(
        left=df[0].reset_index(drop=True),
        right=df[1].reset_index(drop=True),
        left_index=True,
        right_index=True
    )
    df.columns = ['rebalance', 'most_recent']
    df = df.query('rebalance >= @self.start')
    # df = df.query('rebalance_date >= "2023-01-01"')
    df.reset_index(drop=True, inplace=True)

    return df


def save_output(self):
    # Identificação dos fatores
    if len(self.factor_list.keys()) == 1:
        factor = list(self.factor_list)[0]

        if self.custom_name is not None:    # Se existir custom_name, usa esse nome
            subfactor = self.custom_name
        elif len(self.factor_list[factor]) == 1:    # Se só existir um bucket, chama de bucket
            subfactor = list(self.factor_list[factor])[0]
        else:   # Se forem múltiplos buckets, chama de composite
            subfactor = 'composite'

    else:
        factor = 'multi_factor'
        subfactor = '_'.join(list(self.factor_list)) if self.custom_name is None else self.custom_name

    # Se usar "factor_relevance"...
    universe = list(self.investment_universe.values())[0]
    if self.use_factor_relevance is False and self.use_sector_score is False:
        f_path = fr"\\xpdocs\Research\Equities\XP Data\factor_strategies\members\{factor}\{subfactor}-unconstrained-{universe}.parquet"
    else:
        f_path = fr"\\xpdocs\Research\Equities\XP Data\factor_strategies\members\{factor}\{subfactor}-{universe}.parquet"

    df = self.raw_data.copy()
    df.to_parquet(f_path)

    return


def check_file(self):
    universe = list(self.investment_universe.values())[0]

    # Identificação dos fatores
    if len(self.factor_list.keys()) == 1:
        factor = list(self.factor_list)[0]

        if self.custom_name is not None:
            subfactor = self.custom_name
        elif len(self.factor_list[factor]) == 1:
            subfactor = self.factor_list[factor][0]
        else:
            subfactor = 'composite'

    else:
        factor = 'multi_factor'
        subfactor = '_'.join(list(self.factor_list)) if self.custom_name is None else self.custom_name

    f_path = r"\\xpdocs\Research\Equities\XP Data\factor_strategies\members"
    f_path += f"\\{factor}\\{subfactor}-{universe}.pkl"

    return f_path, os.path.isfile(f_path)
