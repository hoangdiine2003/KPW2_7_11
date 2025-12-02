import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize, Bounds, LinearConstraint
import streamlit as st

class EfficientFrontierAnalyzer:
    def __init__(self, prices_df, expected_returns=None):
        """
        Efficient Frontier Analyzer
        """
        self.prices_df = prices_df.copy()
        self.tickers = self.prices_df.columns.tolist()
        self.num_assets = len(self.tickers)
        
        # Calculate returns and statistics
        self.returns = self.prices_df.pct_change().dropna()
        self.cov_matrix = self.returns.cov().values
        
        # Expected returns (annualized)
        if expected_returns is None:
            self.expected_returns = self.returns.mean().values * 252
        else:
            self.expected_returns = np.array(expected_returns, dtype=float)
        
        # Validate inputs
        if self.expected_returns.shape[0] != self.num_assets:
            raise ValueError(f"Expected returns length ({self.expected_returns.shape[0]}) must equal number of assets ({self.num_assets})")
        
        # Initialize optimization results
        self.min_var_portfolio = None
        self.max_sharpe_portfolio = None
        self.efficient_frontier = None
    
    def portfolio_return(self, weights):
        """Calculate portfolio expected return"""
        return np.dot(weights, self.expected_returns)
    
    def portfolio_volatility(self, weights):
        """Calculate portfolio volatility"""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    def sharpe_ratio(self, weights, rf=0.0):
        """Calculate Sharpe ratio"""
        ret = self.portfolio_return(weights) - rf
        vol = self.portfolio_volatility(weights)
        return ret / vol if vol > 0 else 0
    
    def find_minimum_variance_portfolio(self):
        """Find the global minimum variance portfolio using trust-constr"""
        # Constraints and bounds
        bounds = Bounds(0, 1)  # All weights between 0 and 1
        linear_constraint = LinearConstraint(np.ones(self.num_assets), 1, 1)  # Sum of weights = 1
        
        # Initial weights (equal allocation)
        w0 = np.ones(self.num_assets) / self.num_assets
        
        # Objective: minimize volatility
        def objective(weights):
            return self.portfolio_volatility(weights)
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='trust-constr',
            constraints=linear_constraint,
            bounds=bounds
        )
        
        if not result.success:
            raise RuntimeError(f"Min variance optimization failed: {result.message}")
        
        weights = result.x
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        
        self.min_var_portfolio = {
            'weights': weights,
            'return': ret,
            'volatility': vol,
            'sharpe': self.sharpe_ratio(weights),
            'variance': vol ** 2
        }
        
        return self.min_var_portfolio
    
    def find_max_sharpe_portfolio(self):
        """Find the maximum Sharpe ratio portfolio"""
        # Constraints and bounds
        bounds = Bounds(0, 1)
        linear_constraint = LinearConstraint(np.ones(self.num_assets), 1, 1)
        
        # Initial weights
        w0 = np.ones(self.num_assets) / self.num_assets
        
        # Objective: minimize 1/Sharpe ratio (equivalent to maximize Sharpe)
        def objective(weights):
            ret = self.portfolio_return(weights)
            vol = self.portfolio_volatility(weights)
            if ret <= 0 or vol <= 0:
                return 1e10  # Large penalty for invalid portfolios
            return vol / ret
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='trust-constr',
            constraints=linear_constraint,
            bounds=bounds
        )
        
        if not result.success:
            # Try with different initial guess
            w0_alt = self.expected_returns / self.expected_returns.sum()
            result = minimize(
                objective,
                w0_alt,
                method='trust-constr',
                constraints=linear_constraint,
                bounds=bounds
            )
        
        if not result.success:
            st.warning("Max Sharpe optimization failed, using equal weights")
            weights = np.ones(self.num_assets) / self.num_assets
        else:
            weights = result.x
        
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        
        self.max_sharpe_portfolio = {
            'weights': weights,
            'return': ret,
            'volatility': vol,
            'sharpe': self.sharpe_ratio(weights),
            'variance': vol ** 2
        }
        
        return self.max_sharpe_portfolio
    
    def build_efficient_frontier(self, num_points=50):
        """Build the efficient frontier using trust-constr optimization"""
        if self.min_var_portfolio is None:
            self.find_minimum_variance_portfolio()
        
        if self.max_sharpe_portfolio is None:
            self.find_max_sharpe_portfolio()
        
        # Define target returns range
        ret_min = self.min_var_portfolio['return']
        ret_max = min(self.expected_returns.max(), self.max_sharpe_portfolio['return'] * 1.2)
        target_returns = np.linspace(ret_min, ret_max, num_points)
        
        # Constraints and bounds
        bounds = Bounds(0, 1)
        
        frontier_weights = []
        frontier_rets = []
        frontier_vols = []
        frontier_sharpe = []
        
        # Use min variance portfolio as starting point
        w_start = self.min_var_portfolio['weights']
        
        for target_ret in target_returns:
            # Constraints: sum(weights) = 1 AND expected_return = target
            A = np.vstack([np.ones(self.num_assets), self.expected_returns])
            lb = np.array([1.0, target_ret])
            ub = np.array([1.0, target_ret])
            constraints = LinearConstraint(A, lb, ub)
            
            # Objective: minimize volatility
            def objective(weights):
                return self.portfolio_volatility(weights)
            
            # Optimize
            result = minimize(
                objective,
                w_start,
                method='trust-constr',
                constraints=constraints,
                bounds=bounds
            )
            
            if result.success:
                weights = result.x
                ret = self.portfolio_return(weights)
                vol = self.portfolio_volatility(weights)
                sharpe = self.sharpe_ratio(weights)
                
                frontier_weights.append(weights)
                frontier_rets.append(ret)
                frontier_vols.append(vol)
                frontier_sharpe.append(sharpe)
                
                # Update starting point for next iteration
                w_start = weights
        
        # Convert to arrays
        frontier_weights = np.array(frontier_weights)
        frontier_rets = np.array(frontier_rets)
        frontier_vols = np.array(frontier_vols)
        frontier_sharpe = np.array(frontier_sharpe)
        
        self.efficient_frontier = {
            'weights': frontier_weights,
            'returns': frontier_rets,
            'volatilities': frontier_vols,
            'sharpe_ratios': frontier_sharpe
        }
        
        return self.efficient_frontier
    
    def get_portfolio_allocations_df(self):
        """Get DataFrame with portfolio allocations for efficient frontier"""
        if self.efficient_frontier is None:
            self.build_efficient_frontier()
        
        rows = []
        for i, (vol, ret, weights, sharpe) in enumerate(zip(
            self.efficient_frontier['volatilities'],
            self.efficient_frontier['returns'],
            self.efficient_frontier['weights'],
            self.efficient_frontier['sharpe_ratios']
        )):
            row = {
                'Portfolio': i + 1,
                'Risk (Volatility)': vol,
                'Return': ret,
                'Sharpe Ratio': sharpe
            }
            for j, ticker in enumerate(self.tickers):
                row[f'{ticker} (%)'] = weights[j] * 100
            rows.append(row)
        
        df = pd.DataFrame(rows).sort_values(by='Risk (Volatility)').reset_index(drop=True)
        return df
    
    def plot_efficient_frontier(self):
        """Plot the efficient frontier with Plotly"""
        if self.efficient_frontier is None:
            self.build_efficient_frontier()
        
        # Generate random portfolios for comparison
        n_random = 2000
        random_weights = np.random.random((n_random, self.num_assets))
        random_weights = random_weights / random_weights.sum(axis=1)[:, np.newaxis]
        
        random_returns = []
        random_vols = []
        random_sharpe = []
        
        for weights in random_weights:
            ret = self.portfolio_return(weights)
            vol = self.portfolio_volatility(weights)
            sharpe = self.sharpe_ratio(weights)
            random_returns.append(ret)
            random_vols.append(vol)
            random_sharpe.append(sharpe)
        
        # Create the plot
        fig = go.Figure()
        
        # Add random portfolios
        fig.add_trace(go.Scatter(
            x=random_vols,
            y=random_returns,
            mode='markers',
            marker=dict(
                size=4,
                color=random_sharpe,
                colorscale='Viridis',
                opacity=0.6,
                colorbar=dict(title="Sharpe Ratio")
            ),
            name='Random Portfolios',
            text=[f'Sharpe: {s:.3f}' for s in random_sharpe],
            hovertemplate='<b>Random Portfolio</b><br>Risk: %{x:.4f}<br>Return: %{y:.4f}<br>%{text}<extra></extra>'
        ))
        
        # Add efficient frontier
        fig.add_trace(go.Scatter(
            x=self.efficient_frontier['volatilities'],
            y=self.efficient_frontier['returns'],
            mode='lines',
            line=dict(color='red', width=3),
            name='Efficient Frontier',
            hovertemplate='<b>Efficient Portfolio</b><br>Risk: %{x:.4f}<br>Return: %{y:.4f}<extra></extra>'
        ))
        
        # Add minimum variance portfolio
        if self.min_var_portfolio:
            fig.add_trace(go.Scatter(
                x=[self.min_var_portfolio['volatility']],
                y=[self.min_var_portfolio['return']],
                mode='markers',
                marker=dict(
                    size=15,
                    color='black',
                    symbol='star',
                    line=dict(width=2, color='white')
                ),
                name=f"Min Variance (Sharpe: {self.min_var_portfolio['sharpe']:.3f})",
                hovertemplate='<b>Min Variance Portfolio</b><br>Risk: %{x:.4f}<br>Return: %{y:.4f}<extra></extra>'
            ))
        
        # Add maximum Sharpe portfolio
        if self.max_sharpe_portfolio:
            fig.add_trace(go.Scatter(
                x=[self.max_sharpe_portfolio['volatility']],
                y=[self.max_sharpe_portfolio['return']],
                mode='markers',
                marker=dict(
                    size=15,
                    color='gold',
                    symbol='star',
                    line=dict(width=2, color='red')
                ),
                name=f"Max Sharpe (Sharpe: {self.max_sharpe_portfolio['sharpe']:.3f})",
                hovertemplate='<b>Max Sharpe Portfolio</b><br>Risk: %{x:.4f}<br>Return: %{y:.4f}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Đường Biên Hiệu Quả (Efficient Frontier) - Markowitz',
            xaxis_title='Risk (Volatility)',
            yaxis_title='Expected Return',
            hovermode='closest',
            showlegend=False,
            width=900,
            height=600
        )
        
        return fig
    
    def get_investment_recommendation(self, investment_amount=1000000, portfolio_type='min_var'):
        """Get investment recommendation based on selected portfolio type"""
        if portfolio_type == 'min_var':
            if self.min_var_portfolio is None:
                self.find_minimum_variance_portfolio()
            portfolio = self.min_var_portfolio
            portfolio_name = "Minimum Variance"
        elif portfolio_type == 'max_sharpe':
            if self.max_sharpe_portfolio is None:
                self.find_max_sharpe_portfolio()
            portfolio = self.max_sharpe_portfolio
            portfolio_name = "Maximum Sharpe"
        else:
            raise ValueError("portfolio_type must be 'min_var' or 'max_sharpe'")
        
        # Calculate investment amounts
        recommendations = []
        for i, ticker in enumerate(self.tickers):
            weight = portfolio['weights'][i]
            amount = investment_amount * weight
            recommendations.append({
                'Mã CP': ticker,
                'Tỷ trọng (%)': weight * 100,
                'Số tiền đầu tư (VND)': amount,
                'Số tiền đầu tư (triệu VND)': amount / 1_000_000
            })
        
        df_recommendation = pd.DataFrame(recommendations)
        df_recommendation = df_recommendation.sort_values('Tỷ trọng (%)', ascending=False).reset_index(drop=True)
        
        return df_recommendation, portfolio, portfolio_name