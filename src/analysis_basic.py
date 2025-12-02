"""
Phân tích thống kê cơ bản cho cổ phiếu
"""

import pandas as pd
import numpy as np
import streamlit as st

def summarize_all_stocks(returns: pd.DataFrame, rf_annual: float = 0.03) -> pd.DataFrame:
    """
    Tạo bảng thống kê tổng quan cho tất cả cổ phiếu
    """
    summary_list = []
    
    for stock in returns.columns:
        stock_returns = returns[stock].dropna()
        
        if len(stock_returns) > 20:  # Cần đủ dữ liệu
            # Tính toán thống kê cơ bản
            mean_daily = stock_returns.mean()
            std_daily = stock_returns.std()
            annual_return = mean_daily * 252
            annual_vol = std_daily * np.sqrt(252)
            
            # Tính Sharpe ratio
            excess_return = annual_return - rf_annual
            sharpe = excess_return / annual_vol if annual_vol > 0 else 0
            
            stats = {
                'Mã CP': stock,
                'Lợi nhuận TB (% năm)': annual_return * 100,
                'Độ lệch chuẩn (% năm)': annual_vol * 100,
                'Sharpe Ratio': sharpe,
                'Số quan sát': len(stock_returns)
            }
            summary_list.append(stats)
    
    if summary_list:
        df = pd.DataFrame(summary_list).set_index('Mã CP')
        return df
    else:
        return pd.DataFrame()

def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Tính Max Drawdown từ tỷ suất lợi nhuận
    """
    try:
        # Tính cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Tính running maximum
        running_max = cum_returns.expanding().max()
        
        # Tính drawdown
        drawdown = (cum_returns - running_max) / running_max
        
        # Trả về max drawdown (số âm)
        return drawdown.min()
    except:
        return 0.0

def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Tính Max Drawdown từ tỷ suất lợi nhuận
    """
    try:
        # Tính cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Tính running maximum
        running_max = cum_returns.expanding().max()
        
        # Tính drawdown
        drawdown = (cum_returns - running_max) / running_max
        
        # Trả về max drawdown (số âm)
        return drawdown.min()
    except:
        return 0.0

def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Tính hệ số Beta của cổ phiếu so với thị trường
    """
    try:
        # Lấy cột đầu tiên nếu market_returns là DataFrame
        if isinstance(market_returns, pd.DataFrame):
            market_returns = market_returns.iloc[:, 0]
        
        # Đảm bảo cùng index
        common_idx = stock_returns.index.intersection(market_returns.index)
        if len(common_idx) < 10:
            return 1.0  # Beta mặc định
        
        stock_aligned = stock_returns.loc[common_idx]
        market_aligned = market_returns.loc[common_idx]
        
        # Tính Beta = Cov(stock, market) / Var(market)
        covariance = np.cov(stock_aligned, market_aligned)[0, 1]
        market_variance = np.var(market_aligned)
        
        if market_variance > 0:
            beta = covariance / market_variance
        else:
            beta = 1.0
            
        return beta
        
    except Exception:
        return 1.0

def calculate_technical_indicators(prices: pd.Series) -> dict:
    """
    Tính các chỉ báo kỹ thuật cơ bản
    """
    indicators = {}
    
    try:
        # Moving averages
        indicators['SMA_20'] = prices.rolling(20).mean().iloc[-1]
        indicators['SMA_50'] = prices.rolling(50).mean().iloc[-1]
        indicators['EMA_12'] = prices.ewm(span=12).mean().iloc[-1]
        indicators['EMA_26'] = prices.ewm(span=26).mean().iloc[-1]
        
        # MACD
        indicators['MACD'] = indicators['EMA_12'] - indicators['EMA_26']
        
        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        indicators['RSI'] = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Bollinger Bands
        sma_20 = prices.rolling(20).mean()
        std_20 = prices.rolling(20).std()
        indicators['BB_Upper'] = (sma_20 + (std_20 * 2)).iloc[-1]
        indicators['BB_Lower'] = (sma_20 - (std_20 * 2)).iloc[-1]
        indicators['BB_Middle'] = sma_20.iloc[-1]
        
    except Exception as e:
        st.warning(f"Không thể tính chỉ báo kỹ thuật: {str(e)}")
        
    return indicators

def analyze_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Tính ma trận tương quan giữa các cổ phiếu
    """
    return returns.corr()

def calculate_risk_metrics(returns: pd.Series, confidence_level: float = 0.05) -> dict:
    """
    Tính các metrics rủi ro
    """
    metrics = {}
    
    try:
        # VaR (Value at Risk)
        metrics['VaR'] = returns.quantile(confidence_level)
        
        # CVaR (Conditional Value at Risk)
        var_threshold = metrics['VaR']
        tail_losses = returns[returns <= var_threshold]
        metrics['CVaR'] = tail_losses.mean() if len(tail_losses) > 0 else var_threshold
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics['Max_Drawdown'] = drawdown.min()
        
        # Volatility
        metrics['Volatility'] = returns.std()
        
        # Downside Deviation
        negative_returns = returns[returns < 0]
        metrics['Downside_Deviation'] = negative_returns.std() if len(negative_returns) > 0 else 0
        
    except Exception as e:
        st.warning(f"Không thể tính risk metrics: {str(e)}")
        
    return metrics