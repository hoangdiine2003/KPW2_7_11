"""
Portfolio Markowitz Module
Tối ưu hóa danh mục đầu tư theo lý thuyết Markowitz
"""

import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
import streamlit as st
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px

class MarkowitzOptimizer:
    """
    Bộ tối ưu hóa danh mục theo lý thuyết Markowitz
    """
    
    def __init__(self, returns: pd.DataFrame):
        """
        Khởi tạo optimizer
        """
        self.returns = returns.dropna()
        self.n_assets = len(returns.columns)
        self.asset_names = list(returns.columns)
        
        # Tính expected returns và covariance matrix
        self.expected_returns = self.returns.mean() * 252  # Annualized
        
        # Sử dụng Ledoit-Wolf shrinkage estimator
        lw = LedoitWolf()
        cov_matrix, _ = lw.fit(self.returns).covariance_, lw.shrinkage_
        self.cov_matrix = pd.DataFrame(
            cov_matrix * 252,  # Annualized
            index=self.asset_names,
            columns=self.asset_names
        )
    
    def portfolio_stats(self, weights: np.ndarray) -> tuple:
        """
        Tính thống kê danh mục
        """
        portfolio_return = np.dot(weights, self.expected_returns)
        portfolio_var = np.dot(weights.T, np.dot(self.cov_matrix.values, weights))
        portfolio_vol = np.sqrt(portfolio_var)
        
        # Risk-free rate (giả định 3%/năm)
        rf_rate = 0.03
        sharpe_ratio = (portfolio_return - rf_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        return portfolio_return, portfolio_vol, sharpe_ratio
    
    def minimize_risk(self, target_return: float = None) -> dict:
        """
        Tối thiểu hóa rủi ro với ràng buộc tỷ suất lợi nhuận
        """
        # Constraint: sum of weights = 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Constraint: target return if specified
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.dot(x, self.expected_returns) - target_return
            })
        
        # Bounds: each weight between 0 and 1 (long-only)
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess: equal weights
        x0 = np.array([1/self.n_assets] * self.n_assets)
        
        # Objective: minimize portfolio variance
        def objective(weights):
            return np.dot(weights.T, np.dot(self.cov_matrix.values, weights))
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            weights = result.x
            ret, vol, sharpe = self.portfolio_stats(weights)
            
            return {
                'weights': weights,
                'expected_return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'success': True
            }
        else:
            return {'success': False, 'message': result.message}
    
    def maximize_sharpe(self) -> dict:
        """
        Tối đa hóa Sharpe ratio (Tangency portfolio)
        """
        # Constraint: sum of weights = 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Bounds: each weight between 0 and 1
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess
        x0 = np.array([1/self.n_assets] * self.n_assets)
        
        # Objective: minimize negative Sharpe ratio
        rf_rate = 0.03
        def objective(weights):
            ret, vol, _ = self.portfolio_stats(weights)
            if vol > 0:
                return -(ret - rf_rate) / vol
            else:
                return 1000  # Large penalty
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            weights = result.x
            ret, vol, sharpe = self.portfolio_stats(weights)
            
            return {
                'weights': weights,
                'expected_return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'success': True
            }
        else:
            return {'success': False, 'message': result.message}
    
    def efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """
        Tính toán efficient frontier
        """
        # Tìm min và max possible returns
        min_ret_result = self.minimize_risk()
        max_ret = self.expected_returns.max()
        
        if not min_ret_result['success']:
            st.error("Không thể tính efficient frontier")
            return pd.DataFrame()
        
        min_ret = min_ret_result['expected_return']
        
        # Tạo dãy target returns
        target_returns = np.linspace(min_ret, max_ret * 0.95, n_points)
        
        frontier_data = []
        
        for target_ret in target_returns:
            result = self.minimize_risk(target_ret)
            
            if result['success']:
                frontier_data.append({
                    'return': result['expected_return'],
                    'risk': result['volatility'],
                    'sharpe': result['sharpe_ratio'],
                    'weights': result['weights']
                })
        
        return pd.DataFrame(frontier_data)
    
    def plot_efficient_frontier(self) -> go.Figure:
        """
        Vẽ biểu đồ efficient frontier
        """
        frontier_df = self.efficient_frontier()
        
        if frontier_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="Không thể tính efficient frontier",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
        
        # Tìm portfolio tối ưu
        max_sharpe_result = self.maximize_sharpe()
        min_risk_result = self.minimize_risk()
        
        fig = go.Figure()
        
        # Efficient frontier
        fig.add_trace(go.Scatter(
            x=frontier_df['risk'] * 100,
            y=frontier_df['return'] * 100,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='blue', width=2)
        ))
        
        # Maximum Sharpe portfolio
        if max_sharpe_result['success']:
            fig.add_trace(go.Scatter(
                x=[max_sharpe_result['volatility'] * 100],
                y=[max_sharpe_result['expected_return'] * 100],
                mode='markers',
                name='Max Sharpe',
                marker=dict(color='red', size=12, symbol='star')
            ))
        
        # Minimum variance portfolio
        if min_risk_result['success']:
            fig.add_trace(go.Scatter(
                x=[min_risk_result['volatility'] * 100],
                y=[min_risk_result['expected_return'] * 100],
                mode='markers',
                name='Min Risk',
                marker=dict(color='green', size=12, symbol='diamond')
            ))
        
        # Individual assets
        asset_risks = np.sqrt(np.diag(self.cov_matrix)) * 100
        asset_returns = self.expected_returns * 100
        
        fig.add_trace(go.Scatter(
            x=asset_risks,
            y=asset_returns,
            mode='markers+text',
            name='Cổ phiếu',
            text=self.asset_names,
            textposition="top center",
            marker=dict(color='orange', size=8)
        ))
        
        fig.update_layout(
            title='Đường Biên Hiệu Quả (Efficient Frontier)',
            xaxis_title='Rủi ro (% năm)',
            yaxis_title='Lợi nhuận kỳ vọng (% năm)',
            hovermode='closest',
            showlegend=False,
            height=600
        )
        
        return fig
    
    def analyze_portfolio(self, weights: np.ndarray) -> dict:
        """
        Phân tích chi tiết một danh mục cụ thể
        """
        ret, vol, sharpe = self.portfolio_stats(weights)
        
        # Risk contribution
        portfolio_var = np.dot(weights.T, np.dot(self.cov_matrix.values, weights))
        marginal_contrib = np.dot(self.cov_matrix.values, weights)
        contrib = weights * marginal_contrib / portfolio_var
        
        analysis = {
            'weights': dict(zip(self.asset_names, weights)),
            'expected_return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'var_95': np.percentile(
                np.dot(self.returns.values, weights), 5
            ) * np.sqrt(252),  # Annual VaR
            'risk_contribution': dict(zip(self.asset_names, contrib))
        }
        
        return analysis

@st.cache_data
def get_optimal_portfolios(returns_data: pd.DataFrame) -> dict:
    """
    Tính các danh mục tối ưu chính
    """
    optimizer = MarkowitzOptimizer(returns_data)
    
    portfolios = {}
    
    # Maximum Sharpe Ratio Portfolio
    max_sharpe = optimizer.maximize_sharpe()
    if max_sharpe['success']:
        portfolios['max_sharpe'] = max_sharpe
    
    # Minimum Variance Portfolio
    min_variance = optimizer.minimize_risk()
    if min_variance['success']:
        portfolios['min_variance'] = min_variance
    
    # Equal Weight Portfolio (benchmark)
    n_assets = len(returns_data.columns)
    equal_weights = np.array([1/n_assets] * n_assets)
    ret, vol, sharpe = optimizer.portfolio_stats(equal_weights)
    
    portfolios['equal_weight'] = {
        'weights': equal_weights,
        'expected_return': ret,
        'volatility': vol,
        'sharpe_ratio': sharpe,
        'success': True
    }
    
    return portfolios