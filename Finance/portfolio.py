import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

EASTERN_TIME = 'America/New_York'
TODAY = pd.Timestamp.now(EASTERN_TIME).date()
YTD = [f'{TODAY.year}-01-01', TODAY]
SQRT_252 = np.sqrt(252)

class Portfolio:

    def __init__(self, equities, cash, benchmark_indices=['^GSPC', '^DJI', '^IXIC'], period=YTD):
        self.equities = equities
        self.cash = cash
        self.benchmark_indices = benchmark_indices
        self.period = period

        # download equity close prices, scale by shares, and calculate metrics for stock price simulation
        # volatility here is rolling standard deviation of log returns

        self.equities_df = yf.download(tickers=list(equities), start=period[0], end=period[1], auto_adjust=True, progress=False)['Close'].ffill()
        self.equities_df.columns.name = None
        self.equities_df.index.name = None
        for ticker in equities:
            self.equities_df[ticker] *= equities[ticker]
            self.equities_df[f'{ticker} Daily Returns'] = self.equities_df[ticker].pct_change()
            self.equities_df[f'{ticker} Log Returns'] = np.log1p(self.equities_df[f'{ticker} Daily Returns'])
            self.equities_df[f'{ticker} ROI'] = np.expm1(self.equities_df[f'{ticker} Log Returns'].cumsum())
            self.equities_df[f'{ticker} Volatility'] = self.equities_df[f'{ticker} Log Returns'].rolling(window=21).std()

        # total value of portfolio is the sum of close prices of each equity held plus cash
        # calculate metrics to measure portfolio performance and compare to benchmark indices

        self.performance_df = pd.DataFrame(index=self.equities_df.index, dtype='float64')
        self.performance_df['Portfolio'] = self.equities_df[list(equities)].sum(axis=1) + cash
        self.performance_df['Daily Returns'] = self.performance_df['Portfolio'].pct_change()
        self.performance_df['Log Returns'] = np.log1p(self.performance_df['Daily Returns'])
        self.performance_df['ROI'] = np.expm1(self.performance_df['Log Returns'].cumsum())
        self.performance_df['Volatility'] = self.performance_df['Log Returns'].rolling(window=21).std()

        # download benchmark indices (S&P 500, Dow, and Nasdaq by default) and calculate same metrics as for the portfolio

        self.benchmark_df = yf.download(tickers=benchmark_indices, start=period[0], end=period[1], auto_adjust=True, progress=False)['Close'].ffill()
        self.benchmark_df.columns.name = None
        self.benchmark_df.index.name = None
        for ticker in benchmark_indices:
            self.benchmark_df[f'{ticker} Daily Returns'] = self.benchmark_df[ticker].pct_change()
            self.benchmark_df[f'{ticker} Log Returns'] = np.log1p(self.benchmark_df[f'{ticker} Daily Returns'])
            self.benchmark_df[f'{ticker} ROI'] = np.expm1(self.benchmark_df[f'{ticker} Log Returns'].cumsum())
            self.benchmark_df[f'{ticker} Volatility'] = self.benchmark_df[f'{ticker} Log Returns'].rolling(window=21).std()

    def plot_benchmark(self, metric, center_ylim=False, scale=1):
        """Used for comparing metric between portfolio and benchmark indices, with options to center the y-axis and scale values. Plots metric in new figure. Set scale equal to SQRT_252 to annualize."""
        plt.figure(metric)
        plt.plot(self.performance_df[metric] * scale, label='Portfolio')
        for ticker in self.benchmark_indices:
            plt.plot(self.benchmark_df[f'{ticker} {metric}'] * scale, label=ticker)
        if center_ylim:
            max_ylim = np.abs(plt.ylim()).max()
            plt.ylim(-max_ylim, max_ylim)
            plt.axhline(0, color='black', linestyle='--', zorder=0)
        plt.legend()

    def plot_correlation(self):
        """Used for visualizing the correlation between portfolio, equities, and benchmark indices. Plots correlation heatmap in new figure."""
        plt.figure('Correlation Heatmap')
        sns.heatmap(pd.concat([self.performance_df['Portfolio'], self.equities_df[list(self.equities)], self.benchmark_df[self.benchmark_indices]], axis=1).corr(), annot=True, cmap='coolwarm', center=0)
        plt.tight_layout()

    def simulate(self, trials, days, stochastic_volatility=False, plot=False):
        """Simulates portfolio performance. Uses geometric brownian motion to model each equity's price over the given number of days.
        Setting stochastic_volatility equal to true models volatility using geometric brownian motion as well, to then be used in simulating prices.
        Setting plot equal to true plots the simulated portfolios in one figure and confidence bands in another. Returns simulated portfolios."""

        # create multiindex to prevent dataframe fragmentation, then create dataframe for the simulation of each equity's log returns and price
        # create dataframe for total portfolio value, returned by function

        multiindex = pd.MultiIndex.from_product([np.arange(1, trials + 1), np.append(np.char.add(list(self.equities), ' Log Returns'), list(self.equities))])
        simulation_df = pd.DataFrame(index=pd.bdate_range(np.datetime64(self.period[1]) + np.timedelta64(1, 'D'), periods=days).date, columns=multiindex, dtype='float64')
        portfolios_df = pd.DataFrame(index=simulation_df.index, columns=np.arange(1, trials + 1), dtype='float64')
        vol_of_vol_ser = np.log1p(self.equities_df[np.char.add(list(self.equities), ' Volatility')].pct_change()).std().set_axis(self.equities)
        mean_log_returns_ser = self.equities_df[np.char.add(list(self.equities), ' Log Returns')].mean().set_axis(self.equities)
        for x in range(1, trials + 1):
            for ticker in self.equities:

                # geometric brownian motion for volatility (if stochastic_volatility set to True) and log returns, which are then cumulatively summed to simulate price movements

                volatility = self.equities_df[f'{ticker} Volatility'].iat[-1] * np.exp(vol_of_vol_ser[ticker] * np.random.normal(0, 1, days) * stochastic_volatility)
                simulation_df[x, f'{ticker} Log Returns'] = mean_log_returns_ser[ticker] + volatility * np.random.normal(0, 1, days)
                simulation_df[x, ticker] = self.equities_df[ticker].iat[-1] * np.exp(simulation_df[x][f'{ticker} Log Returns'].cumsum())
            portfolios_df[x] = simulation_df[x][list(self.equities)].sum(axis=1) + self.cash

        if plot:
            plt.figure('Simulated Portfolios')
            plt.plot(portfolios_df)
            plt.figure('Confidence Bands')
            ci_99 = int(0.005 * trials)
            ci_95 = int(0.025 * trials)
            ci_90 = int(0.05 * trials)
            plt.fill_between(portfolios_df.index, np.partition(portfolios_df, kth=ci_99, axis=1)[:, ci_99], np.partition(portfolios_df, kth=-ci_99, axis=1)[:, -ci_99], alpha=0.2, label='99%')
            plt.fill_between(portfolios_df.index, np.partition(portfolios_df, kth=ci_95, axis=1)[:, ci_95], np.partition(portfolios_df, kth=-ci_95, axis=1)[:, -ci_95], alpha=0.2, label='95%')
            plt.fill_between(portfolios_df.index, np.partition(portfolios_df, kth=ci_90, axis=1)[:, ci_90], np.partition(portfolios_df, kth=-ci_90, axis=1)[:, -ci_90], alpha=0.2, label='90%')
            plt.plot(portfolios_df.mean(axis=1), color='black', label='Mean')
            plt.legend()

        return portfolios_df


if __name__ == '__main__':
    p = Portfolio({
        'NVDA'  : 7.99,
        'AAPL'  : 6.47,
        'MSFT'  : 6.24,
        'AMZN'  : 4.23,
        'GOOGL' : 2.84
        }, cash=100.00)

    p.plot_benchmark('ROI', center_ylim=True)
    p.plot_benchmark('Volatility', scale=SQRT_252)
    p.plot_correlation()
    p.simulate(trials=1000, days=100, stochastic_volatility=True, plot=True)
    plt.show()
