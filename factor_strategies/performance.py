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


def get_performance(returns: pd.Series) -> pd.Series:
    """Calculate performance from returns."""
    performance = np.cumprod(1 + returns)
    return performance


def calculate_drawdown(cum_returns: pd.Series) -> pd.Series:
    """Calculate drawdown from cumulative returns."""
    drawdown = (cum_returns / cum_returns.cummax() - 1)
    return drawdown


def calculate_returns(screening: pd.DataFrame, weighting_scheme: str) -> pd.DataFrame:
    df = screening[['code', 'date', 'overall_score', 'overall_quantile']].copy()
    ativos = list(df['code'].unique())
    min_date = df['date'].min()
    max_quantile = df['overall_quantile'].max()

    prices = get_prices()
    volatility = get_volatility()

    df = pd.merge(left=df, right=volatility, left_on=['code', 'date'], right_index=True, how='left')

    returns = prices.query('code == @ativos & date >= @min_date')
    returns = returns.pivot_table(index='date', columns='code', values='price_close')
    returns = returns.pct_change().fillna(0)

    factor_returns = pd.DataFrame()
    for i in range(1, max_quantile + 1):
        w = df.query(f'overall_quantile == {i}')

        if weighting_scheme == 'equal_weighted':
            w = w.pivot_table(index='date', columns='code', values='overall_score')
            w = w.notnull().astype('int')
        elif weighting_scheme == 'inverse_volatility':
            w = w.pivot_table(index='date', columns='code', values='3m_volatility')
            w = w.pow(-1).fillna(0)

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
    """Calculate information ratios from returns."""
    if excess_returns:
        benchmark_returns = get_benchmark(benchmark='br_ibovespa')
        returns_mask = pd.merge(
            left=returns,
            right=benchmark_returns,
            how='left',
            left_index=True,
            right_index=True
        )
        returns_mask = returns_mask.sub(returns_mask['benchmark_returns'], axis=0)
        returns_mask.drop(columns='benchmark_returns', inplace=True)
        returns_mask.loc[:, returns_mask.filter(like='long_short').columns] = returns.loc[:, returns.filter(like='long_short').columns]
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


def calculate_hit_ratios(cum_returns: pd.Series, frequency: str = 'M') -> pd.Series:
    """Calculate hit ratios from cumulative returns."""
    df = cum_returns.resample(frequency).last().pct_change()
    return (df > 0).sum() / len(df)


def calculate_annualized_returns(cum_returns: pd.Series) -> pd.Series:
    """Calculate annualized returns from cumulative returns."""
    days_held = (cum_returns.index[-1] - cum_returns.index[0]).days
    ann_returns = (cum_returns.iloc[-1] / cum_returns.iloc[0]) ** (252 / days_held) - 1
    return ann_returns


def calculate_information_coefficient(screening: pd.DataFrame, prices: pd.DataFrame) -> tuple:
    """Calculate information coefficient."""
    df = pd.merge(prices.assign(fwd_close_price=prices.groupby('code')['price_close'].shift(-21))
                  .assign(fwd_1m_ret=lambda x: x.fwd_close_price / x.price_close - 1),
                  screening, on=['code', 'date'], how='right')
    res = stats.spearmanr(df['overall_score'], df['fwd_1m_ret'], nan_policy='omit')
    return [res.correlation, res.pvalue], df