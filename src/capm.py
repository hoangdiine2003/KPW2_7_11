"""
CAPM Module
Mô hình định giá tài sản vốn (Capital Asset Pricing Model) và SML
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from scipy import stats

class CAPMAnalyzer:
    """
    Phân tích CAPM cho cổ phiếu
    """
    
    def __init__(self, stock_returns: pd.DataFrame, market_returns: pd.Series, rf_rate: float = 0.03):
        """
        Khởi tạo CAPM analyzer
        """
        self.stock_returns = stock_returns
        self.market_returns = market_returns
        self.rf_rate = rf_rate
        self.rf_daily = rf_rate / 252  # Convert to daily
        
        # Align data
        self._align_data()
        
        # Calculate excess returns
        self.stock_excess_returns = self.stock_returns_aligned - self.rf_daily
        self.market_excess_returns = self.market_returns_aligned - self.rf_daily
    
    def _align_data(self):
        """Căn chỉnh dữ liệu cổ phiếu và thị trường"""
        # Chuẩn hóa index về dạng ngày (YYYY-MM-DD), loại bỏ timezone
        def _normalize_index(idx: pd.Index) -> pd.Index:
            try:
                # Convert to pandas Timestamps then normalize to date (midnight)
                ts = pd.to_datetime(idx)
                # Remove timezone if present
                if getattr(ts, "tz", None) is not None:
                    ts = ts.tz_convert(None)
                # Normalize to date only to avoid time mismatches
                ts = ts.normalize()
                return ts
            except Exception:
                # Fallback: try to parse as strings
                return pd.to_datetime(pd.Index(idx)).normalize()

        sr = self.stock_returns.copy()
        mr = self.market_returns.copy()

        # Ensure sorted indices and no duplicates
        sr = sr.sort_index()
        mr = mr.sort_index()

        sr.index = _normalize_index(sr.index)
        mr.index = _normalize_index(mr.index)

        # Nếu market là Series, đảm bảo tên cột nhất quán khi cần
        # Tìm giao nhau theo ngày
        common_idx = sr.index.intersection(mr.index)

        # Chỉ giữ các ngày chung, loại bỏ NaN toàn hàng
        self.stock_returns_aligned = sr.loc[common_idx].dropna(how="all")
        self.market_returns_aligned = mr.loc[common_idx].dropna()

        # Trong trường hợp index vẫn không khớp hoàn toàn, reindex market theo stock để đảm bảo căn chỉnh
        self.market_returns_aligned = self.market_returns_aligned.reindex(self.stock_returns_aligned.index).dropna()

        # Đồng bộ lại stock theo market (sau khi market có thể bị drop thêm do NaN)
        self.stock_returns_aligned = self.stock_returns_aligned.loc[self.market_returns_aligned.index]
    
    def calculate_beta(self, stock_symbol: str) -> dict:
        """
        Tính Beta cho một cổ phiếu
        
        Args:
            stock_symbol: Mã cổ phiếu
        
        Returns:
            Dictionary chứa beta và thống kê liên quan
        """
        if stock_symbol not in self.stock_returns_aligned.columns:
            return {'error': f'Không tìm thấy {stock_symbol}'}
        
        # Get clean data
        y = self.stock_returns_aligned[stock_symbol].dropna()
        x = self.market_returns_aligned.reindex(y.index).dropna()
        
        # Align again after dropna
        common_idx = y.index.intersection(x.index)
        if len(common_idx) < 20:  # Need minimum observations
            return {'error': 'Không đủ dữ liệu (cần ít nhất 20 quan sát)'}
            
        y_clean = y.loc[common_idx]
        x_clean = x.loc[common_idx]
        
        # Calculate excess returns for regression
        y_excess = y_clean - self.rf_daily
        x_excess = x_clean - self.rf_daily
        
        try:
            # Linear regression using numpy for more stable results
            x_vals = x_excess.values.reshape(-1, 1)
            y_vals = y_excess.values
            
            # Calculate beta using covariance method (more stable)
            covariance = np.cov(x_excess, y_excess)[0, 1]
            variance = np.var(x_excess)
            
            if variance == 0:
                return {'error': 'Phương sai thị trường bằng 0'}
                
            beta = covariance / variance
            alpha = np.mean(y_excess) - beta * np.mean(x_excess)
            
            # Calculate R-squared
            y_pred = alpha + beta * x_excess
            ss_res = np.sum((y_excess - y_pred) ** 2)
            ss_tot = np.sum((y_excess - np.mean(y_excess)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Standard error calculations
            residuals = y_excess - y_pred
            mse = np.mean(residuals**2)
            x_var = np.var(x_excess)
            se_beta = np.sqrt(mse / (len(x_excess) * x_var)) if x_var > 0 else 0
            
            # T-statistic and p-value
            t_stat = beta / se_beta if se_beta > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(x_excess) - 2)) if len(x_excess) > 2 else 1
            
            return {
                'beta': float(beta),
                'alpha': float(alpha * 252),  # Annualized
                'r_squared': float(max(0, min(1, r_squared))),  # Ensure 0-1 range
                'std_error': float(se_beta),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'observations': int(len(x_excess))
            }
            
        except Exception as e:
            return {'error': f'Lỗi tính toán: {str(e)}'}
    
    def calculate_all_betas(self) -> pd.DataFrame:
        """
        Tính Beta cho tất cả cổ phiếu
        """
        results = []
        
        for stock in self.stock_excess_returns.columns:
            beta_stats = self.calculate_beta(stock)
            
            if 'error' not in beta_stats:
                beta_stats['stock'] = stock
                results.append(beta_stats)
            else:
                st.warning(f"Không thể tính Beta cho {stock}: {beta_stats['error']}")
        
        if results:
            df = pd.DataFrame(results).set_index('stock')
            return df
        else:
            return pd.DataFrame()
    
    def expected_return_capm(self, beta: float) -> float:
        """
        Tính tỷ suất lợi nhuận kỳ vọng theo CAPM
        """
        market_premium = self.market_excess_returns.mean() * 252
        return self.rf_rate + beta * market_premium
    
    def plot_security_market_line(self) -> go.Figure:
        """
        Vẽ Security Market Line (SML)
        """
        betas_df = self.calculate_all_betas()
        
        if betas_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="Không thể tính Beta cho các cổ phiếu",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Calculate actual returns (annualized)
        actual_returns = self.stock_returns_aligned.mean() * 252 * 100  # Convert to %
        
        # SML line
        beta_range = np.linspace(0, betas_df['beta'].max() * 1.2, 100)
        sml_returns = [self.expected_return_capm(b) * 100 for b in beta_range]
        
        fig = go.Figure()
        
        # SML line
        fig.add_trace(go.Scatter(
            x=beta_range,
            y=sml_returns,
            mode='lines',
            name='Security Market Line',
            line=dict(color='red', width=2)
        ))
        
        # Individual stocks
        stock_betas = []
        stock_actual_returns = []
        stock_names = []
        
        for stock in betas_df.index:
            if stock in actual_returns.index:
                stock_betas.append(betas_df.loc[stock, 'beta'])
                stock_actual_returns.append(actual_returns.loc[stock])
                stock_names.append(stock)
        
        fig.add_trace(go.Scatter(
            x=stock_betas,
            y=stock_actual_returns,
            mode='markers+text',
            name='Cổ phiếu',
            text=stock_names,
            textposition="top center",
            marker=dict(color='blue', size=10)
        ))
        
        # Risk-free rate point
        fig.add_trace(go.Scatter(
            x=[0],
            y=[self.rf_rate * 100],
            mode='markers',
            name='Lãi suất phi rủi ro',
            marker=dict(color='green', size=12, symbol='diamond')
        ))
        
        # Risk-free rate horizontal line
        fig.add_hline(
            y=self.rf_rate * 100,
            line_dash="dash",
            line_color="green",
            line_width=2,
            annotation_text=f"RF = {self.rf_rate * 100:.1f}%",
            annotation_position="top right",
            annotation=dict(
                font=dict(size=12, color="green"),
                bgcolor="white",
                bordercolor="green"
            )
        )
        
        fig.update_layout(
            title='Security Market Line (SML)',
            xaxis_title='Beta',
            yaxis_title='Tỷ suất lợi nhuận kỳ vọng (%)',
            hovermode='closest',
            showlegend=False,
            height=600
        )
        
        # Add SML formula annotation
        fig.add_annotation(
            x=0.98, y=0.02,
            xref="paper", yref="paper",
            text="SML: E(Ri) = RF + βi × [E(RM) - RF]",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=12, family="Arial")
        )
        
        return fig
    
    def plot_beta_regression(self, stock_symbol: str) -> go.Figure:
        """
        Vẽ biểu đồ hồi quy Beta cho một cổ phiếu
        
        Args:
            stock_symbol: Mã cổ phiếu
        
        Returns:
            Plotly figure
        """
        if stock_symbol not in self.stock_excess_returns.columns:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Không tìm thấy dữ liệu cho {stock_symbol}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Get data
        y = self.stock_excess_returns[stock_symbol].dropna()
        x = self.market_excess_returns.reindex(y.index).dropna()
        
        common_idx = y.index.intersection(x.index)
        y_clean = y.loc[common_idx] * 100  # Convert to %
        x_clean = x.loc[common_idx] * 100  # Convert to %
        
        # Calculate beta
        beta_stats = self.calculate_beta(stock_symbol)
        
        if 'error' in beta_stats:
            fig = go.Figure()
            fig.add_annotation(
                text=beta_stats['error'],
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Regression line
        x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
        y_pred = beta_stats['alpha'] * 100 + beta_stats['beta'] * x_range
        
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=x_clean,
            y=y_clean,
            mode='markers',
            name='Dữ liệu quan sát',
            marker=dict(color='blue', opacity=0.6)
        ))
        
        # Regression line
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_pred,
            mode='lines',
            name=f'Đường hồi quy (β = {beta_stats["beta"]:.3f})',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f'Hồi quy CAPM - {stock_symbol}',
            xaxis_title='Tỷ suất lợi nhuận thị trường (%)',
            yaxis_title=f'Tỷ suất lợi nhuận {stock_symbol} (%)',
            showlegend=False,
            annotations=[
                dict(
                    x=0.02, y=0.98,
                    xref="paper", yref="paper",
                    text=f"β = {beta_stats['beta']:.3f}<br>"
                         f"α = {beta_stats['alpha']:.3f}<br>"
                         f"R² = {beta_stats['r_squared']:.3f}",
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="black"
                )
            ],
            height=600
        )
        
        return fig
    
    def analyze_stock_performance(self, stock_symbol: str) -> dict:
        """
        Phân tích hiệu quả cổ phiếu so với CAPM
        """
        beta_stats = self.calculate_beta(stock_symbol)
        
        if 'error' in beta_stats:
            return beta_stats
        
        # Actual return
        actual_return = self.stock_returns_aligned[stock_symbol].mean() * 252
        
        # Expected return by CAPM
        expected_return = self.expected_return_capm(beta_stats['beta'])
        
        # Jensen's Alpha
        jensen_alpha = actual_return - expected_return
        
        # Treynor Ratio
        excess_return = actual_return - self.rf_rate
        treynor_ratio = excess_return / beta_stats['beta'] if beta_stats['beta'] != 0 else 0
        
        analysis = {
            'beta': beta_stats['beta'],
            'alpha_intercept': beta_stats['alpha'],
            'actual_return': actual_return,
            'expected_return_capm': expected_return,
            'jensen_alpha': jensen_alpha,
            'treynor_ratio': treynor_ratio,
            'r_squared': beta_stats['r_squared'],
            'classification': self._classify_stock(beta_stats['beta'], jensen_alpha)
        }
        
        return analysis
    
    def _classify_stock(self, beta: float, jensen_alpha: float) -> str:
        """Phân loại cổ phiếu dựa trên Beta và Alpha"""
        risk_level = "Thấp" if beta < 1 else "Cao"
        performance = "Tốt" if jensen_alpha > 0 else "Kém"
        
        if beta < 0.7:
            risk_class = "Defensive (Phòng thủ)"
        elif beta > 1.3:
            risk_class = "Aggressive (Tích cực)"
        else:
            risk_class = "Neutral (Trung tính)"
        
        return f"{risk_class} - Hiệu quả {performance}"