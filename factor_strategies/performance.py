import pandas as pd
import numpy as np
from scipy import stats
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')


def get_prices():
    df = pd.read_parquet("https://persevera.s3.sa-east-1.amazonaws.com/factor_zoo.parquet",
                         columns=['code', 'date', 'price_close'])
    return df


def get_benchmark(benchmark: str):
    df = pd.read_parquet(os.path.join(DATA_PATH, "consolidado-indicators.parquet"),
                         filters=[('code', '==', benchmark)])
    df = df.set_index('date')
    df = df.drop(columns='code')
    df['benchmark_returns'] = df['close_price'].pct_change()
    df.drop(columns='close_price', inplace=True)
    return df


def calculate_returns(screening, prices, n):
    df = screening.copy()
    ativos = list(df['code'].unique())
    min_date = df['date'].min()
    bins = np.linspace(0, 1, n + 1)

    returns = (
        prices
        .query('code == @ativos & date >= @min_date')
        .pivot_table(index='date', columns='code', values='price_close')
        .pct_change()
        .fillna(0)
    )

    df['index'] = df.groupby('date')['code'].cumcount() + 1
    df['num_members'] = df.groupby('date')['index'].transform('max')
    df['ratio'] = df['index'] / df['num_members']
    df['quantile'] = pd.cut(df['ratio'], bins=bins, labels=False) + 1

    factor_returns = pd.DataFrame()

    for i in range(1, n + 1):
        w = (
            df.query(f'quantile == {i}')
            .pivot_table(index='date', columns='code', values='overall_score')
            .notnull()
            .astype('int')
        )

        weight_mask = returns[w.columns]
        weight_mask[:] = np.nan
        weight_mask = pd.concat([weight_mask, w])
        weight_mask = weight_mask[~weight_mask.index.duplicated(keep='last')]
        weight_mask = weight_mask.sort_index()
        weight_mask = weight_mask.ffill()
        weight_mask = (weight_mask.T / weight_mask.sum(axis=1)).T

        return_mask = returns[w.columns] * weight_mask
        group_return = return_mask.sum(axis=1)

        factor_returns = pd.merge(
            left=factor_returns,
            right=pd.DataFrame(group_return, columns=[f'rank_{i}']),
            left_index=True,
            right_index=True,
            how='outer'
        )

    factor_returns['long_short'] = factor_returns.iloc[:, 0] - factor_returns.iloc[:, -1]
    return factor_returns


def get_performance(returns):
    returns = returns.copy()
    returns.iloc[0] = np.nan
    performance = np.cumprod(1 + returns)
    performance.iloc[0] = 1
    performance.bfill(inplace=True)
    return performance


def calculate_drawdown(cum_returns):
    drawdown = (cum_returns / cum_returns.cummax() - 1)
    return drawdown


def custom_stock_selection(screening, kind, constraints):
    df = screening.copy()
    possible_scores = ['value_score', 'growth_score', 'momentum_score', 'quality_score', 'risk_score', 'size_score']
    df = df.filter([possible_scores])

    if kind == 'intersection':
        #TODO: calcular os quintiles para cada fator; corr entre score e fwd returns
        i = 1

    return


def calculate_quantile_returns(screening, prices):
    df = screening.copy()
    ativos = list(df['code'].unique())
    min_date = df['date'].min()
    max_quantile = df['overall_quantile'].max()

    returns = (
        prices
        .query('code == @ativos & data >= @min_date')
        .pivot_table(index='date', columns='code', values='price_close')
        .pct_change()
        .fillna(0)
    )

    factor_returns = pd.DataFrame()
    for i in range(1, max_quantile + 1):
        w = (
            df
            .query(f'overall_quantile == {i}')
            .pivot_table(index='date', columns='code', values='overall_score')
            .notnull()
            .astype('int')
        )

        weight_mask = returns[w.columns]
        weight_mask[:] = np.nan
        weight_mask = pd.concat([weight_mask, w])
        weight_mask = weight_mask[~weight_mask.index.duplicated(keep='last')]
        weight_mask = weight_mask.sort_index()
        weight_mask = weight_mask.ffill()
        weight_mask = (weight_mask.T / weight_mask.sum(axis=1)).T

        return_mask = returns[w.columns] * weight_mask
        group_return = return_mask.sum(axis=1)

        factor_returns = pd.merge(
            left=factor_returns,
            right=pd.DataFrame(group_return, columns=[f'rank_{i}']),
            left_index=True,
            right_index=True,
            how='outer'
        )

    factor_returns['long_short'] = factor_returns.iloc[:, 0] - factor_returns.iloc[:, -1]
    return factor_returns


def calculate_information_ratios(returns, excess_returns=True):
    if excess_returns:
        benchmark_returns = get_benchmark(benchmark='Ibovespa')
        returns_mask = pd.merge(
            left=returns,
            right=benchmark_returns,
            how='left',
            left_index=True,
            right_index=True
        )
        returns_mask = returns_mask.sub(returns_mask['benchmark_returns'], axis=0)
        returns_mask.drop(columns='benchmark_returns', inplace=True)
        returns_mask['long_short'] = returns['long_short']
        returns_mask.fillna(0, inplace=True)

        avg_returns = returns_mask.mean()
        std_return = returns_mask.std()
    else:
        avg_returns = returns.mean()
        std_return = returns.std()

    # Annualized returns and std
    average_annualized_returns = (1 + avg_returns) ** 252 - 1
    annualized_std_return = std_return * np.sqrt(252)

    # Compute the Information Ratio
    information_ratio = average_annualized_returns / annualized_std_return

    return information_ratio


def calculate_hit_ratios(cum_returns, frequency='M'):
    df = cum_returns.resample(frequency).last().pct_change()
    greater_then_zero = df.gt(0).sum()
    return greater_then_zero / len(df)


def calculate_annualized_returns(cum_returns):
    d0 = cum_returns.index[0]
    dt = cum_returns.index[-1]
    days_held = (dt - d0).days

    ann_returns = (cum_returns.iloc[-1] / cum_returns.iloc[0]) ** (252 / days_held) - 1
    ann_returns.name = 0
    return ann_returns


def calculate_information_coefficient(screening, prices):
    df = (
        prices
        .assign(fwd_close_price = prices.groupby('code')['price_close'].shift(-21))
        .assign(fwd_1m_ret = lambda x: x.fwd_close_price / x.adj_close_price - 1)
        .merge(right=screening, on=['code', 'date'], how='right')
    )
    
    res = stats.spearmanr(a=df['overall_score'], b=df['fwd_1m_ret'], nan_policy='omit')
    return [res.correlation, res.pvalue], df
