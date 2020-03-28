# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:38:10 2020

@author: Pablo Alvarado
"""

#let's 
import pandas as pd
import numpy as np
from pandas_datareader.data import DataReader
import pandas_datareader.data as web
from datetime import date
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy.stats import norm
import plotly.io as pio
import plotly.express as px
import time 
from tqdm import tqdm
import winsound
from scipy.optimize import minimize
from plotly.subplots import make_subplots

TOLERANCE = 1e-10

pio.renderers.default = "browser"
def c():
    for i in tqdm(range(10)):
        time.sleep(3)

c()
#set the function for returns, mean and volatility
tickers=['KLAC','CINF','TGT','PEP','ZTS','AIZ','LMT','SO','MDT','HD']
wts=[0.125,0.119,0.117,0.106,0.1,0.098,0.096,0.094,0.084,0.061]

start_date1=date(2019,1,1)
end_date1=date(2020,1,1)
initial_investment=1000000
def get_stock_quote(tickers):
    c()
    start=start_date1
    end=end_date1
    source='yahoo'
    price=DataReader(tickers,source,start,end)
    df=pd.DataFrame(price)
    return df

def get_returns():
    c()
    adj_close=get_stock_quote(tickers)['Adj Close']
    rets=adj_close.pct_change()
    rets=rets.dropna()
    rets=pd.DataFrame(rets)
    return rets

def get_mean():
    c()
    returns=get_returns()
    return returns.mean()
def get_std():
    c()
    returns=get_returns()
    return returns.std()
#get the data for risk free rate and benchmark - we will use that later
def get_risk_free_rate_daily():
    c()
    start=start_date1
    end=end_date1
    source_two='fred'
    code='DGS10'
    rfr=DataReader(code,source_two,start,end)
    rfr=rfr.dropna()
    rfr=rfr.mean()/100
    d_rfr=rfr/252
    return np.float64(d_rfr)
#this function is for regression
def get_benchmark():
    c()
    start=start_date1
    end=end_date1
    source='yahoo'
    bench=['^GSPC']
    adj_close=DataReader(bench,source,start,end)['Adj Close']
    bench_returns=adj_close.pct_change()
    bench_returns=bench_returns.dropna()
    return bench_returns
#this function is for sortino
def get_benchmark_average_daily_return():
    c()
    rets=get_benchmark()
    mean=rets.mean()
    return mean
#figure out what is the correlation and covariance of assets
def get_correlation():
    c()
    returns=get_returns()
    correlation=returns.corr()
    return correlation
def get_covariance():
    c()
    returns=get_returns()
    covariance=returns.cov()
    return covariance
#let the fun begin
def simulate_portfolios():
    c()
    port_simulations=100000
    cov_matrix=get_covariance()
    mean_returns=get_mean()
    port_returns=[]
    port_volatility=[]
    stock_weights=[]
    sharpe_ratio=[]
    d_rfr=get_risk_free_rate_daily()
    num_assets=len(tickers)
    for single_port in range(port_simulations):
        weights=np.random.random(num_assets)
        weights/=np.sum(weights)
        returns=np.dot(weights,mean_returns)
        volatility=np.sqrt(np.dot(weights.T,np.dot(cov_matrix,weights)))
        sharpe=float((returns-d_rfr)/volatility)
        port_returns.append(returns)
        port_volatility.append(volatility)
        stock_weights.append(weights)
        sharpe_ratio.append(sharpe)
    portfolio={'Returns':port_returns,
               'Volatility':port_volatility,
               'Sharpe Ratio':sharpe_ratio}
    for counter,symbol in enumerate(tickers):
        portfolio[symbol+' Weight']=[Weight[counter] for Weight in stock_weights]
    df=pd.DataFrame(portfolio)
    return df

def get_best_sharpe_port():
    c()
    all_portfolios=pd.DataFrame(simulate_portfolios())
    max_sharpe_port = all_portfolios.iloc[all_portfolios['Sharpe Ratio'].idxmax()]
    return max_sharpe_port
#risk parity
TOLERANCE = 1e-10
def _allocation_risk(weights, covariances):
    portfolio_risk = np.sqrt((weights * covariances * weights.T))[0, 0]
    return portfolio_risk
def _assets_risk_contribution_to_allocation_risk(weights, covariances):
    portfolio_risk = _allocation_risk(weights, covariances)
    assets_risk_contribution = np.multiply(weights.T, covariances * weights.T) \
        / portfolio_risk
    return assets_risk_contribution


def _risk_budget_objective_error(weights, args):
    covariances = args[0]
    assets_risk_budget = args[1]
    weights = np.matrix(weights)
    portfolio_risk = _allocation_risk(weights, covariances)
    assets_risk_contribution = \
        _assets_risk_contribution_to_allocation_risk(weights, covariances)
    assets_risk_target = \
        np.asmatrix(np.multiply(portfolio_risk, assets_risk_budget))
    error = \
        sum(np.square(assets_risk_contribution - assets_risk_target.T))[0, 0]
    return error


def _get_risk_parity_weights(covariances, assets_risk_budget, initial_weights):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                   {'type': 'ineq', 'fun': lambda x: x})
    optimize_result = minimize(fun=_risk_budget_objective_error,
                               x0=initial_weights,
                               args=[covariances, assets_risk_budget],
                               method='SLSQP',
                               constraints=constraints,
                               tol=TOLERANCE,
                               options={'disp': False})
    weights = optimize_result.x
    return weights

yahoo_tickers=tickers
def get_weights(yahoo_tickers,
                start_date=start_date1,
                end_date=end_date1):
    prices = pd.DataFrame([web.DataReader(t,
                                          'yahoo',
                                          start_date,
                                          end_date).loc[:, 'Adj Close']
                           for t in yahoo_tickers],
                          index=yahoo_tickers).T.asfreq('B').ffill()
    covariances = 52.0 * \
        prices.asfreq('W-FRI').pct_change().iloc[1:, :].cov().values
    assets_risk_budget = [1 / prices.shape[1]] * prices.shape[1]
    init_weights = [1 / prices.shape[1]] * prices.shape[1]
    weights = \
        _get_risk_parity_weights(covariances, assets_risk_budget, init_weights)
    weights = pd.Series(weights, index=prices.columns, name='weight')
    return weights

def get_portfolio_returns():
    c()
    returns=get_returns()
    ap=pd.DataFrame(simulate_portfolios())
    ap=ap.sort_values(by='Sharpe Ratio',ascending=False)
    w=ap[:1]
    w=w.drop(columns=['Returns','Volatility','Sharpe Ratio'])
    w=np.array(w)
    weighted_returns=(w*returns)
    port_ret=weighted_returns.sum(axis=1)
    return port_ret
#just a work with portfolio
def get_mean_portfolio_returns():
    c()
    mu=get_portfolio_returns().mean()
    return mu
def get_std_of_portfolio_returns():
    c()
    sigma=get_portfolio_returns().std()
    return sigma

def get_returns_bp():
    c()
    ret_data=get_returns()
    weighted_returns=(wts*ret_data)
    port_ret=weighted_returns.sum(axis=1)
    return port_ret

def get_mean_bp_returns():
    c()
    mu=get_returns_bp().mean()
    return mu

def get_std_of_bp():
    c()
    std=get_returns_bp().std()
    return std

def get_sharpe_ratio_for_bp():
    c()
    d_rfr=get_risk_free_rate_daily()
    returns_bp=get_returns_bp()
    mean=returns_bp.mean()
    std=returns_bp.std()
    sharpe_ratio=((mean-d_rfr)/std)
    return sharpe_ratio

def get_treynor_ratio_for_bp():
    regression_tab=pd.DataFrame(get_benchmark())
    regression_tab['Portfolio Returns']=get_returns_bp()
    x=regression_tab['^GSPC']
    y=regression_tab['Portfolio Returns']
    model = sm.OLS(x,y).fit()
    beta=model.params[0]
    d_risk_free=get_risk_free_rate_daily()
    treynor_ratio=(y.mean()-d_risk_free)/beta
    return treynor_ratio

def get_sortino_for_bp():
    c()
    d_rfr=get_risk_free_rate_daily()
    tab=pd.DataFrame(get_benchmark())
    tab['Portfolio Returns']=get_returns_bp()
    tab['Portfolio Drawdown Risk']=np.where(tab['Portfolio Returns']<tab['^GSPC'].mean(),
       tab['Portfolio Returns'],
       tab['^GSPC'].mean())
    downside_risk=tab['Portfolio Drawdown Risk'].std()
    sortino_ratio=(((tab['Portfolio Returns'].mean())-d_rfr)/downside_risk)
    return sortino_ratio

#some additional ratios
def get_sharpe_ratio():
    d_rfr=get_risk_free_rate_daily()
    mean=get_mean_portfolio_returns()
    std=get_std_of_portfolio_returns()
    ratio=(mean-d_rfr)/std
    return ratio

def get_sharpe_ratio_rp():
    d_rfr=get_risk_free_rate_daily()
    returns_rp=get_risk_parity_returns()
    mean=returns_rp.mean()
    std=returns_rp.std()
    ratio=(mean-d_rfr)/std
    return ratio

def get_treynor_ratio():
    c()
    regression_tab=pd.DataFrame(get_benchmark())
    regression_tab['Portfolio Returns']=get_portfolio_returns()
    x=regression_tab['^GSPC']
    y=regression_tab['Portfolio Returns']
    model = sm.OLS(x,y).fit()
    beta=model.params[0]
    d_risk_free=get_risk_free_rate_daily()
    treynor_ratio=(y.mean()-d_risk_free)/beta
    return treynor_ratio

def get_risk_parity_returns():
    c()
    returns=get_returns()
    weights=pd.DataFrame(get_weights(tickers))
    weights=weights['weight'].to_list()
    portfolio_returns=(returns*weights).sum(axis=1)
    return portfolio_returns
def get_risk_parity_mean_returns():
    c()
    returns=get_returns()
    weights=pd.DataFrame(get_weights(tickers))
    weights=weights['weight'].to_list()
    portfolio_returns=(returns*weights).sum(axis=1)
    mean=portfolio_returns.mean()
    return mean
def get_risk_parity_std_returns():
    c()
    returns=get_returns()
    weights=pd.DataFrame(get_weights(tickers))
    weights=weights['weight'].to_list()
    portfolio_returns=(returns*weights).sum(axis=1)
    std=portfolio_returns.std()
    return std

def get_treynor_for_rp_port():
    c()
    regression_tab=pd.DataFrame(get_benchmark())
    returns=get_returns()   
    weights=pd.DataFrame(get_weights(tickers))
    weights=weights['weight'].to_list()
    portfolio_returns=(returns*weights).sum(axis=1)
    regression_tab['Portfolio Returns']=portfolio_returns
    x=regression_tab['^GSPC']
    y=regression_tab['Portfolio Returns']
    model=sm.OLS(x,y).fit()
    beta=model.params[0]
    d_risk_free=get_risk_free_rate_daily()
    treynor_ratio=(y.mean()-d_risk_free)/beta
    return treynor_ratio

def get_sortino():
    c()
    d_rfr=get_risk_free_rate_daily()
    tab=pd.DataFrame(get_benchmark())
    tab['Portfolio Returns']=get_portfolio_returns()
    tab['Portfolio Drawdown Risk']=np.where(tab['Portfolio Returns']<tab['^GSPC'].mean(),
       tab['Portfolio Returns'],
       tab['^GSPC'].mean())
    downside_risk=tab['Portfolio Drawdown Risk'].std()
    sortino_ratio=(((tab['Portfolio Returns'].mean())-d_rfr)/downside_risk)
    return sortino_ratio

def get_sortino_for_rpp():
    c()
    d_rfr=get_risk_free_rate_daily()
    tab=pd.DataFrame(get_benchmark())
    tab['Portfolio Returns']=get_risk_parity_returns()
    tab['Portfolio Drawdown Risk']=np.where(tab['Portfolio Returns']<tab['^GSPC'].mean(),
       tab['Portfolio Returns'],
       tab['^GSPC'].mean())
    downside_risk=tab['Portfolio Drawdown Risk'].std()
    sortino_ratio=(((tab['Portfolio Returns'].mean())-d_rfr)/downside_risk)
    return sortino_ratio

#value at risk
#we need to get cumulative return
def get_cumulative_return_of_each_stock():
    returns=get_returns()
    cumulative_return=((returns+1).cumprod())-1
    df=pd.DataFrame(cumulative_return)
    return df

def get_cumulative_return_of_portfolio():
    c()
    ret=get_portfolio_returns()
    cumulative_return_of_p=((ret+1).cumprod())-1
    df=pd.DataFrame({'cumulative return of portfolio':cumulative_return_of_p})
    return df

def get_cumulative_return_of_benchmark():
    bench_ret=get_benchmark()
    cr_of_bench=((bench_ret+1).cumprod())-1
    df=pd.DataFrame(cr_of_bench)
    return df

def get_cumulative_return_of_risk_parity():
    c()
    returns=get_returns()
    weights=pd.DataFrame(get_weights(tickers))
    weights=weights['weight'].to_list()
    portfolio_returns=(returns*weights).sum(axis=1)
    cumulative_portfolio_returns=((portfolio_returns+1).cumprod())-1
    df=pd.DataFrame({'risk parity portfolio':cumulative_portfolio_returns})
    return df

def get_cumulative_return_of_bp():
    c()
    ret=get_returns_bp()
    cumulative_return_of_p=((ret+1).cumprod())-1
    df=pd.DataFrame({'cumulative return of bank portfolio':cumulative_return_of_p})
    return df
#if you wanna know what are the sums you might be loosing
def calculate_value_at_risk():
    c()
    port_mean=get_mean_portfolio_returns()
    port_stdev=get_std_of_portfolio_returns()
    mean_investment=(1+port_mean)*initial_investment
    stdev_investment=initial_investment*port_stdev
    conf_level1=.05
    cutoff1=norm.ppf(conf_level1,mean_investment,stdev_investment)
    var_1d1=initial_investment-cutoff1
    return np.round(var_1d1,2)
def calculate_value_at_risk_for_rp():
    c()
    port_mean=get_risk_parity_mean_returns()
    port_stdev=get_risk_parity_std_returns()
    mean_investment=(1+port_mean)*initial_investment
    stdev_investment=initial_investment*port_stdev
    conf_level1=.05
    cutoff1=norm.ppf(conf_level1,mean_investment,stdev_investment)
    var_1d1=initial_investment-cutoff1
    return np.round(var_1d1,2)
def calculate_value_at_risk_for_bp():
    c()
    port_mean=get_mean_bp_returns()
    port_stdev=get_std_of_bp()
    mean_investment=(1+port_mean)*initial_investment
    stdev_investment=initial_investment*port_stdev
    conf_level1=.05
    cutoff1=norm.ppf(conf_level1,mean_investment,stdev_investment)
    var_1d1=initial_investment-cutoff1
    return np.round(var_1d1,2)
def calculate_var_bp_in_15_days():
    c()
    var_array=[]
    days_counter=[]
    days=int(15)
    orig_var=calculate_value_at_risk_for_bp()
    for i in range(1,days+1):
        var_array.append(np.round(orig_var*np.sqrt(i),2))
        days_counter.append(np.array(i))
        print(str(i)+'day VaR at 95% confidence:'+str(np.round(orig_var*np.sqrt(i),2)))
    df=pd.DataFrame(var_array, columns=['value'])
    df['day']=days_counter
    return df
    
#the table with values and etc that we will try to plot
def calculate_value_at_risk_in_15_days():
    c()
    var_array=[]
    days_counter=[]
    days=int(15)
    orig_var=calculate_value_at_risk()
    for i in range(1,days+1):
        var_array.append(np.round(orig_var*np.sqrt(i),2))
        days_counter.append(np.array(i))
        print(str(i)+'day VaR at 95% confidence:'+str(np.round(orig_var*np.sqrt(i),2)))
    df=pd.DataFrame(var_array, columns=['value'])
    df['day']=days_counter
    return df
def calculate_value_at_risk_in_15_days_rpp():
    c()
    var_array=[]
    days_counter=[]
    days=int(15)
    orig_var=calculate_value_at_risk_for_rp()
    for i in range(1,days+1):
        var_array.append(np.round(orig_var*np.sqrt(i),2))
        days_counter.append(np.array(i))
        print(str(i)+'day VaR at 95% confidence:'+str(np.round(orig_var*np.sqrt(i),2)))
    df=pd.DataFrame(var_array,columns=['value'])
    df['day']=days_counter
    return df
"""the plots"""

def plot_mu_sigma_space():
    mu=get_mean()
    sigma=get_std()
    tab={'ticker':tickers,
         'average return':mu,
         'volatility':sigma}
    df=pd.DataFrame(tab)
    fig=px.scatter(df,x='volatility',y='average return',
                   hover_data=['ticker'],title='Risk VS Return Scatterplot')
    return fig.show()

def plot_cumulative_return_of_portfolio():
    df=get_cumulative_return_of_portfolio()
    fig=px.line(df,y='cumulative return of portfolio', title='Cumulative Return of Portfolio')
    return fig.show()

def plot_efficient_frontier():
    c()
    df=pd.DataFrame(simulate_portfolios())
    fig=px.scatter(df,x='Volatility', y='Returns',
                   title='Efficient Frontier')
    return fig.show()

def plot_var():
    data=calculate_value_at_risk_in_15_days()
    fig=px.line(data,x='day',y='value',title='Value at risk')
    return fig.show()

def plot_correlation_matrix():
    c()
    corr=pd.DataFrame(get_correlation())
    fig=px.imshow(corr,title='correlation matrix')
    return fig.show()
#3D plots
def plot_3d_efficient_frontier():
    c()
    df=pd.DataFrame(simulate_portfolios())
    fig=px.scatter_3d(df,x='Volatility',y='Returns',z='Sharpe Ratio',
                      title='3D Efficient Frontier')
    return fig.show()
def plot_all_cumulative_returns():
    portfolio=pd.DataFrame(get_cumulative_return_of_portfolio())
    portfolio['Risk Parity Portfolio']=pd.DataFrame(get_cumulative_return_of_risk_parity())
    portfolio['Bank Portfolio']=pd.DataFrame(get_cumulative_return_of_bp())
    portfolio['Market Returns']=pd.DataFrame(get_cumulative_return_of_benchmark())
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=portfolio.index,y=portfolio['cumulative return of portfolio'], name='Highest Sharpe Ratio Portfolio Cumulative Return'))
    fig.add_trace(go.Scatter(x=portfolio.index,y=portfolio['Risk Parity Portfolio'],name='Risk Parity Portfolio Cumulative Return'))
    fig.add_trace(go.Scatter(x=portfolio.index,y=portfolio['Bank Portfolio'],name='Bank Portfolio Cumulative Return'))
    fig.add_trace(go.Scatter(x=portfolio.index,y=portfolio['Market Returns'],name='Market Cumulative Return'))
    return fig.show()

#initial settings to get the summary. No worries, it is super slow.
sharpe_ratio = round(get_sharpe_ratio(),4)
sharpe_ratio_rp=round(get_sharpe_ratio_rp(),4)
sharpe_ratio_bp=round(get_sharpe_ratio_for_bp(),4)
treynor_ratio=round(get_treynor_ratio(),4)
treynor_ratio_rp=round(get_treynor_for_rp_port(),4)
treynor_ratio_bp=round(get_treynor_ratio_for_bp(),4)
sortino_ratio=round(get_sortino(),4)
sortino_ratio_rp=round(get_sortino_for_rpp(),4)
sortino_ratio_bp=round(get_sortino_for_bp(),4)
var=pd.DataFrame(calculate_value_at_risk_in_15_days())
var_rpp=pd.DataFrame(calculate_value_at_risk_in_15_days_rpp())
var_bp=pd.DataFrame(calculate_var_bp_in_15_days())
eff_front_df=pd.DataFrame(simulate_portfolios())

fig=go.Figure(data=[go.Table(
        header=dict(values=['Index','Sharpe Ratio Portfolio','Risk Parity Portfolio']),
        cells=dict(values=[['sharpe ratio','treynor ratio','sortino ratio'],[sharpe_ratio,treynor_ratio,sortino_ratio],[sharpe_ratio_rp,treynor_ratio_rp,sortino_ratio_rp]]))])
#GRAND FINALEEEEE
#define variables for the dashboard
portfolio=pd.DataFrame(get_cumulative_return_of_portfolio())
portfolio['Risk Parity Portfolio']=pd.DataFrame(get_cumulative_return_of_risk_parity())
portfolio['Bank Portfolio']=pd.DataFrame(get_cumulative_return_of_bp())
portfolio['Market Returns']=pd.DataFrame(get_cumulative_return_of_benchmark())
eff_front_df=pd.DataFrame(simulate_portfolios())
#define the main frame (what the dash is going to look like)
fig = make_subplots(
        rows=2,cols=2,
        specs=[[{'type':'scatter'},{'type':'table'}],
               [{'type':'scatter'},{'type':'scatter3d'}]],
               subplot_titles=("Cumulative returns for portfolios","Summary table", "Projected value at risk","3D Efficient frontier"))
#adding traces to the main frame
fig.add_trace(go.Scatter(x=portfolio.index,y=portfolio['cumulative return of portfolio'],
                         name='Highest Sharpe Portfolio Cumulative Return'),
              row=1,col=1)
fig.add_trace(go.Scatter(x=portfolio.index,y=portfolio['Risk Parity Portfolio'],
                         name='Risk Parity Portfolio Cumulative Return'),
    row=1,col=1)
fig.add_trace(go.Scatter(x=portfolio.index,y=portfolio['Market Returns'], 
                         name='Market Cumulative Return'), 
    row=1,col=1)
fig.add_trace(go.Scatter(x=portfolio.index,y=portfolio['Bank Portfolio'],
                         name='Bank Portfolio Cumulative Return'),
    row=1,col=1)
fig.add_trace(go.Table(
        header=dict(values=['Index','Sharpe Ratio Portfolio','Risk Parity Portfolio','Bank Portfolio']),
        cells=dict(values=[['sharpe ratio','treynor ratio','sortino ratio'],
                           [sharpe_ratio,treynor_ratio,sortino_ratio],
                           [sharpe_ratio_rp,treynor_ratio_rp,sortino_ratio_rp],
                           [sharpe_ratio_bp,treynor_ratio_bp,sortino_ratio_bp]])), 
    row=1,col=2)
fig.add_trace(go.Scatter3d(x=eff_front_df['Volatility'],y=eff_front_df['Returns'],z=eff_front_df['Sharpe Ratio'],
                           name='3D Efficient Frontier',mode='markers'),
    row=2,col=2)
fig.add_trace(go.Scatter(x=var['day'],y=var['value'],
                         name='Value at risk 15 days projections for best sharpe ratio portfolio'), 
    row=2,col=1)
fig.add_trace(go.Scatter(x=var_rpp['day'],y=var['value'],
                         name='Value at risk 15 days profections for risk parity portfolio'),
    row=2,col=1)
fig.add_trace(go.Scatter(x=var_bp['day'],y=var['value'],
                         name='Value at risk 15 days projections for bank portfolio'),
    row=2,col=1)
#make the dash look better
fig.update_xaxes(title_text='Date',row=1,col=1)
fig.update_yaxes(title_text='Cumulative Return, %',row=1,col=1)
fig.update_xaxes(title_text='Day projected',row=2,col=1)
fig.update_yaxes(title_text='Value at risk,$',row=2,col=1)
fig.update_xaxes(title_text='Risk',row=2,col=2)
fig.update_yaxes(title_text='Return',row=2,col=2)
fig.update_layout(width = 1500, height= 1000, title_text="Portfolio Summary")
fig.show()
#it takes long to execute, so I made a sound to signal execution
duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)
#that's it.