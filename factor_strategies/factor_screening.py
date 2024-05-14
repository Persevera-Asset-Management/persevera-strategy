import pandas as pd
import numpy as np
import logging
import time
import json
import os
from datetime import datetime

import utils, performance


class MultiFactorStrategy:

    def __init__(self, factor_list: dict, custom_name: object, sector_filter: str, sector_comparison: str,
                 sector_specifics: str, holding_period: int, start: str, business_day: int, use_buckets: bool,
                 use_factor_relevance: bool, use_sector_score: bool, market_cap_neutralization: bool,
                 outlier_percentile: list, investment_universe: dict, memory: bool, freq: str, quantile: int):
        self.factor_relevance = None
        self.db = None
        self.data_mkt = None
        self.rebalance_interval = None
        self.raw_data = None
        self.outlier_percentile = outlier_percentile
        self.factor_list = factor_list
        self.custom_name = custom_name
        self.sector_filter = sector_filter
        self.sector_comparison = sector_comparison
        self.sector_specifics = sector_specifics
        self.holding_period = holding_period
        self.freq = freq
        self.start = start
        self.business_day = business_day
        self.use_buckets = use_buckets
        self.use_factor_relevance = use_factor_relevance
        self.use_sector_score = use_sector_score
        self.market_cap_neutralization = market_cap_neutralization
        self.investment_universe = investment_universe
        self.quantile = quantile
        self.memory = memory
        self.report_factor = '_'.join(list(self.factor_list))

        # Definindo o Logging 
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s.%(msecs)03d: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    def initialize(self):
        # Inicia o cronômetro
        start_overall = time.time()

        # Importa os dados base para cada iteração
        self.data_mkt = utils.get_market_data()

        # Importa a importância de cada fator
        self.factor_relevance = utils.get_factor_relevance(self)

        # Calcula as datas dos rebalanceamentos
        if self.freq != 'D':
            self.rebalance_interval = utils.get_rebalance_interval(self, business_day=self.business_day)
        else:
            dates = np.unique(self.data_mkt['date'])
            df_dates = pd.DataFrame([dates, dates]).T
            df_dates.columns = ['rebalance', 'most_recent']
            self.rebalance_interval = (df_dates
                                       .query('rebalance >= @self.start')
                                       .reset_index(drop=True))

        # Prepara e otimiza o factor zoo
        self.db = utils.prepare_factor_zoo(self)

        # Filtra o zoo para setores específicos
        if self.sector_filter is not None:
            self.db.query('sector_specifics == @self.sector_filter', inplace=True)

        logging.info(f'Tempo para inicialização da base: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_overall))}')
        return

    def get_constituents(self, date):
        logging.info(f'Definindo os membros do índice para {format(date, "%Y-%m-%d")}...')

        # Definições dos fatores
        factor_file = open(os.path.join(os.path.dirname(__file__), "factor_definition.json"))
        factor_definition = json.load(factor_file)

        # Cria um Dataframe com os cálculos dos fatores
        df_measures = self.db.query('date == @date')
        df_measures.reset_index(drop=True, inplace=True)

        # Renomeia a coluna das métricas
        df_signals = pd.DataFrame([(factor, factor_group, signal) for factor, factor_groups in self.factor_list.items()
                                   for factor_group, signals in factor_groups.items() for signal in signals],
                                  columns=['factor', 'factor_group', 'signal'])
        df_signals['new_signal'] = pd.Series(dtype='object')

        for i in range(len(df_signals)):
            try:
                col = df_signals.iloc[i]['signal']
                col_factor = df_signals.iloc[i]['factor']
                col_hilo = 'high' if factor_definition[col]['higher_better'] else 'low'
                df_signals.at[i, 'new_signal'] = f'{col_factor}_{col_hilo}_{col}'
            except KeyError:
                pass

        df_measures = df_measures.rename(columns=dict(df_signals[['signal', 'new_signal']].values))
        self.checkpoint_one = df_measures.copy()

        # Reordena as colunas
        static_cols = ['code', 'date', 'size_segment', 'sector_comparison', 'sector_specifics',
                       '21d_median_dollar_volume_traded']
        if list(self.investment_universe)[0] == 'index': static_cols.remove('size_segment')

        static_cols.extend(self.factor_relevance.columns)
        cols = df_measures.drop(static_cols, axis=1, errors='ignore').columns
        cols = sorted(cols)
        factor_cols = cols.copy()
        for col in list(reversed(static_cols)):
            cols.insert(0, col)
        df_measures = df_measures[cols]

        self.checkpoint_two = df_measures.copy()

        # Trata os outliers
        logging.info('Tratando os outliers...')
        for col in factor_cols:
            max_outlier = df_measures[col].quantile(q=self.outlier_percentile[1])
            min_outlier = df_measures[col].quantile(q=self.outlier_percentile[0])
            df_measures[f'{col}_denoised'] = df_measures[col].clip(lower=min_outlier, upper=max_outlier)

        factor_cols = [col for col in df_measures.columns if '_denoised' in col]

        self.checkpoint_three = df_measures.copy()

        # Calcula os scores individuais
        logging.info('Calculando o score de cada ação...')
        # df_measures = utils.get_zscores(self, df_measures, factor_cols)
        df_measures = utils.get_percentiles(df_measures, factor_cols)

        self.checkpoint_four = df_measures.copy()

        # Se flag 'factor_relevance', ignora as métricas irrelevantes
        if self.use_factor_relevance:
            logging.info('Ignorando fatores irrelevantes...')
            rank_cols = [col for col in df_measures.columns if ('denoised' in col) & ('rank' in col)]
            relevance_cols = [col for col in df_measures.columns if ('relevance' in col)]
            metric_cols = [col.replace('relevance_', '') for col in relevance_cols]

            for col in metric_cols:
                rank_cols_ref = [c1 for c1 in rank_cols if col in c1]

                for score_col in rank_cols_ref:
                    df_measures.loc[pd.isnull(df_measures[f'relevance_{col}']), score_col] = np.nan

        self.checkpoint_five = df_measures.copy()

        # Calcula o score por bucket (se True)
        if self.use_buckets:
            logging.info('Calculando o score dos buckets...')
            cols_bucket = []

            for factor in self.factor_list.keys():
                for sub_factor in self.factor_list[factor].keys():
                    bucket_metrics = self.factor_list[factor][sub_factor]
                    cols_bucket.extend([f'bucket_{factor}_{sub_factor}_score'])

                    # Busca o nome das colunas, cujos percentis estão associados ao bucket
                    cols_absolute_bucket = []
                    cols_relative_bucket = []
                    for metric in bucket_metrics:
                        cols_absolute_bucket.extend(
                            [col for col in df_measures.columns if ('absolute_rank' in col) and (metric in col)])
                        cols_relative_bucket.extend(
                            [col for col in df_measures.columns if ('relative_rank' in col) and (metric in col)])

                    # Calcula a média das métricas do bucket
                    df_measures[f'bucket_{factor}_{sub_factor}_absolute_score'] = df_measures.filter(
                        cols_absolute_bucket).mean(axis=1)
                    df_measures[f'bucket_{factor}_{sub_factor}_relative_score'] = df_measures.filter(
                        cols_relative_bucket).mean(axis=1)
                    df_measures[f'bucket_{factor}_{sub_factor}_score'] = df_measures.filter(
                        [f'bucket_{factor}_{sub_factor}_absolute_score', f'bucket_{factor}_{sub_factor}_relative_score']
                    ).mean(axis=1)

        self.checkpoint_six = df_measures.copy()

        # Calcula o score por bucket e fator (EW)
        cols_factor_scores = []
        dict_buckets = dict(map(lambda i: (list(self.factor_list)[i], []), range(len(list(self.factor_list)))))

        if self.use_buckets:
            for factor in list(dict_buckets):
                dict_buckets[factor] = [i for i in cols_bucket if factor in i]

        for factor in list(self.factor_list):
            if self.use_buckets:
                df_measures[f'{factor}_score'] = df_measures.filter(dict_buckets[factor]).mean(axis=1)
            else:
                cols_absolute = [col for col in df_measures.columns if ('absolute_rank' in col) and (factor in col)]
                cols_relative = [col for col in df_measures.columns if ('relative_rank' in col) and (factor in col)]

                df_measures[f'{factor}_absolute_score'] = df_measures.filter(cols_absolute).mean(axis=1)
                df_measures[f'{factor}_relative_score'] = df_measures.filter(cols_relative).mean(axis=1)

                if self.use_sector_score:
                    df_measures[f'{factor}_score'] = (df_measures[f'{factor}_absolute_score'] + df_measures[
                        f'{factor}_relative_score']) / 2
                else:
                    df_measures[f'{factor}_score'] = df_measures[f'{factor}_absolute_score']

            cols_factor_scores.append(f'{factor}_score')

        # Calcula o score combinado dos fatores (EW)
        df_measures['overall_score'] = df_measures[cols_factor_scores].mean(axis=1)
        df_measures.dropna(subset=['overall_score'], inplace=True)  # Se não tiver nenhuma medida, deleta do screening
        df_measures.sort_values(by='overall_score', ascending=False, inplace=True)

        # Retira as ações com menos de 50% das medidas
        # df_measures['count_scores'] = df_measures[cols].count(axis=1)
        # count_treshold = df_measures['count_scores'].max() * .25
        # df_measures = df_measures.query('count_scores >= @count_treshold')

        if self.memory: df_measures.set_index('code').to_excel(
            fr'\\xpdocs\Research\Equities\XP Data\factor_strategies\memory\{datetime.today().strftime("%Y-%m-%d-%H%M%S%f")}-{self.report_factor}.xlsx',
            sheet_name=format(date, "%Y-%m-%d")
        )

        logging.info(f'Screening final composto de {len(df_measures)} ações.')

        return df_measures

    def historical_members(self, save=True, memory=False, how='append'):
        """
        how: 'overwrite', 'append'
        """

        # Inicia o cronômetro
        start_overall = time.time()

        # Checa se já existe um arquivo do fator
        # logging.info('Checando se já há um registro existente...')
        # filename, file_exists = utils.check_file(self)
        # if file_exists:
        #     logging.info('Registro encontrado.')
        #
        #     if how == 'append':
        #         with open(filename, 'rb') as handle:
        #             b = pickle.load(handle)
        #         self.start = b[0].index.max() + timedelta(days=1)
        #         logging.info(self.start)
        #     elif how == 'overwrite':
        #         pass

        # Define a lista de resultados
        self.raw_data = pd.DataFrame()

        if self.market_cap_neutralization:
            for universe in [{'index': 'SMLL'}, {'index': 'MLCX'}]:
                self.investment_universe = universe

                # Re-prepara o factor zoo com o universo atualizado
                self.db = utils.prepare_factor_zoo(self)

                # Itera cada rebalanceamento
                for index, row in self.rebalance_interval.iterrows():
                    start = time.time()

                    dt = row['rebalance']
                    dt_most_recent = row['most_recent']

                    if dt > self.db['date'].max():
                        break

                    constituents = self.get_constituents(dt)

                    # Armazena os dados brutos
                    self.raw_data = pd.concat([self.raw_data, constituents], ignore_index=True)

                    logging.info(f'Tempo parcial: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start))}')

            self.investment_universe = {'index': 'ALL'}

        else:
            # Itera cada rebalanceamento
            for index, row in self.rebalance_interval.iterrows():
                start = time.time()

                dt = row['rebalance']
                dt_most_recent = row['most_recent']

                if dt > self.db['date'].max():
                    break

                constituents = self.get_constituents(dt)

                # Armazena os dados brutos
                self.raw_data = pd.concat([self.raw_data, constituents], ignore_index=True)

                logging.info(f'Tempo parcial: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start))}')

        self.raw_data.sort_values(['date', 'overall_score'], ascending=[True, False], ignore_index=True, inplace=True)

        # # Incluindo na composição existente
        # if len(self.rebalance_interval) == 0:
        #     logging.info('Data do próximo rebalanceamento ainda não disponível.')
        # else:
        #     if how == 'append':
        #         for i in range(len(w)):
        #             w[i] = pd.concat([w[i], b[i]])
        #             w[i].sort_index(inplace=True)
        #
        #             logging.info(w[i])
        #
        #     # Armazena a performance, se save=True
        #     if save is True: save_output(self, w, folder_name='members')

        # data = {'raw_data': self.raw_data, 'composition': w}

        # Calcula os quantiles, se quantile not None
        if isinstance(self.quantile, int):
            self.raw_data['index'] = self.raw_data.groupby('date')['code'].cumcount() + 1
            self.raw_data['num_members'] = self.raw_data.groupby('date')['index'].transform('max')
            self.raw_data.drop(columns=['index'], inplace=True)

            factor_list = list(self.factor_list)
            factor_list.append('overall')
            for factor in factor_list:
                self.raw_data[f'{factor}_rank'] = self.raw_data.groupby('date')[f'{factor}_score'].rank(ascending=False)
                self.raw_data[f'{factor}_ratio'] = self.raw_data[f'{factor}_rank'] / self.raw_data['num_members']
                self.raw_data[f'{factor}_quantile'] = pd.cut(self.raw_data[f'{factor}_ratio'], bins=self.quantile,
                                                             labels=False) + 1
                self.raw_data.drop(columns=[f'{factor}_rank', f'{factor}_ratio'], inplace=True)

            self.raw_data.drop(columns=['num_members'], inplace=True)

        # Armazena a performance, se save=True
        if save is True:
            logging.info('Salvando composições na pasta...')
            utils.save_output(self)

        # Calcula o tempo total
        logging.info(f'Tempo total: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_overall))}')

        return


if __name__ == "__main__":
    f = MultiFactorStrategy(
        factor_list={
            # 'quality': {
            #     # 'earnings_stability': [
            #     #     'sales_variability',
            #     #     'earnings_variability',
            #     #     'roe_variability',
            #     #     'gross_margin_variability',
            #     #     'ebitda_margin_variability',
            #     # ],
            #     'profitability': [
            #         'gross_margin',
            #         'ebitda_margin',
            #         'roic',
            #         'roe',
            #         'fcff_to_ebitda',
            #     ],
            #     'leverage': [
            #         'netdebt_to_ebitda',
            #         'netdebt_to_equity',
            #     ],
            #     'delta_quality': [
            #         'gross_margin_growth',
            #         'ebitda_margin_growth',
            #         'roic_growth',
            #         'roe_growth',
            #     ],
            #     # 'banking_quality': [
            #     #     'npl_to_total_loans',
            #     #     'operating_leverage',
            #     #     'efficiency_ratio',
            #     #     'tier_one_capital_ratio',
            #     #     'fee_revenue_pct',
            #     #     'loans_to_deposits',
            #     # ]
            # },
            # 'value': {
            #     'cross_sectional_value': [
            #         'earnings_yield_ltm',
            #         'ebitda_yield_ltm',
            #         'fcf_yield_ltm',
            #         # 'book_yield_ltm'
            #     ],
            #     'timeseries_value': [
            #         'earnings_yield_10y_percentile',
            #         'ebitda_yield_10y_percentile',
            #         # 'book_yield_10y_percentile'
            #     ],
            # },
            'momentum': {
                'price_momentum': [
                    '12m_momentum',
                    '6m_momentum'
                ],
                # 'lagged_price_momentum': ['12m1_momentum', '6m1_momentum'],
                # 'price_range': ['price_range']
            },
            # 'risk': {
            #     # 'low_risk': ['6m_volatility', '12m_volatility', '36m_volatility', 'beta'],
            #     'low_volatility': [
            #         '9m_volatility',
            #         '36m_volatility'
            #     ],
            #     # 'low_beta': ['beta'],
            # },
            # 'short_interest': {
            #     'short_interest': [
            #         'lending_rate',
            #         'days_to_cover',
            #     ],
            # },
        },
        custom_name=None,
        sector_filter=None,
        sector_comparison='sector_layer_1',
        sector_specifics='sector_layer_1',
        holding_period=1,
        freq='M',
        start='2008-01-01',
        business_day=-2,
        use_buckets=True,
        use_factor_relevance=True,
        use_sector_score=True,
        market_cap_neutralization=False,
        investment_universe={'liquidity_thresh': 0.4, 'liquidity_lookback': 63, 'size_segment': 'ALL'},
        quantile=5,
        memory=False,
        outlier_percentile=[0.02, 0.98],
    )

    f.initialize()
    f.historical_members(save=False, how='overwrite')
    x = f.raw_data.query('date == "2024-04-29"').set_index('code')

    screening = f.raw_data

    prices = performance.get_prices()
    returns = performance.calculate_returns(screening, prices, 5)
    cum_returns = performance.get_performance(returns)
