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
    """Load benchmark data."""
    df = pd.read_parquet(os.path.join(DATA_PATH, "consolidado-indicators.parquet"), filters=[('code', '==', benchmark)])
    if df.empty:
        return df
    df = df.set_index('date').drop(columns='code')
    df['benchmark_returns'] = df['value'].pct_change().dropna()
    return df.drop(columns='value')


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
    """Calculate cumulative performance from returns."""
    return np.cumprod(1 + returns)


def calculate_drawdown(cum_returns):
    """Calculate drawdown from cumulative returns."""
    return cum_returns / cum_returns.cummax() - 1


def calculate_information_ratios(returns, excess_returns=True):
    """Calculate information ratios from returns."""
    if excess_returns:
        benchmark_returns = get_benchmark('br_ibovespa')
        if benchmark_returns.empty:
            return pd.Series()
        returns = adjust_for_benchmark(returns, benchmark_returns)
    return calculate_ratios(returns)


def adjust_for_benchmark(returns, benchmark_returns):
    """Adjust returns for the benchmark."""
    returns = pd.merge(returns, benchmark_returns, left_index=True, right_index=True, how='left')
    returns = returns.sub(returns['benchmark_returns'], axis=0).drop(columns='benchmark_returns').fillna(0)
    return returns


def calculate_ratios(returns):
    """Calculate average and standard deviation of returns."""
    avg_returns = returns.mean()
    std_returns = returns.std()
    annualized_avg_returns = (1 + avg_returns) ** 252 - 1
    annualized_std_returns = std_returns * np.sqrt(252)
    return annualized_avg_returns / annualized_std_returns


def calculate_hit_ratios(cum_returns, frequency='M'):
    """Calculate hit ratios from cumulative returns."""
    df = cum_returns.resample(frequency).last().pct_change()
    return (df > 0).mean()


def calculate_annualized_returns(cum_returns):
    """Calculate annualized returns from cumulative returns."""
    days_held = (cum_returns.index[-1] - cum_returns.index[0]).days
    return (cum_returns.iloc[-1] / cum_returns.iloc[0]) ** (252 / days_held) - 1


def calculate_information_coefficient(screening, prices):
    """Calculate information coefficient."""
    df = pd.merge(
        prices.assign(fwd_close_price=prices.groupby('code')['price_close'].shift(-21))
        .assign(fwd_1m_ret=lambda x: x['fwd_price_close'] / x['price_close'] - 1),
        screening,
        on=['code', 'date'],
        how='right'
    )
    res = stats.spearmanr(df['overall_score'], df['fwd_1m_ret'], nan_policy='omit')
    return [res.correlation, res.pvalue], df
