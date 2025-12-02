"""
Ứng dụng Streamlit - Phân tích Cổ phiếu Việt Nam
Đồ án Tốt nghiệp - Khoa học Dữ liệu Tài chính
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.optimize import minimize

# Import modules
from src.data_loader import *
from src.analysis_basic import *

# Page config
st.set_page_config(
    page_title="Phân tích Cổ phiếu VN",
    page_icon="📈",
    layout="wide"
)

st.title("ỨNG DỤNG PHÂN TÍCH CỔ PHIẾU VIỆT NAM")
st.markdown("*Báo cáo bài tập nhóm cuối kỳ - Khai phá Web | Ứng dụng hỗ trợ ra quyết định đầu tư cổ phiếu*")

with st.sidebar:
    st.header("Thiết lập")
    
    # Stock selection
    symbols_input = st.text_input(
        "Nhập mã cổ phiếu (phân cách bằng dấu phẩy):",
        value="FPT,VNM,HPG,MWG,VCB"
    )
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Từ ngày:",
            value=datetime.now() - timedelta(days=1095),  # 3 years
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "Đến ngày:",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    # Risk-free rate
    rf_annual = st.slider("Lãi suất phi rủi ro (%/năm):", 0.0, 10.0, 3.0) / 100

# Load data
if symbols:
    try:
        with st.spinner("Đang tải dữ liệu..."):
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # Get stock prices
            prices = get_prices_for_list(symbols, start_str, end_str)
            returns = compute_returns(prices)
            
            # Get market data
            mkt_close, mkt_returns = get_market_index_returns(start_str, end_str)
        
        if prices.empty:
            st.error("Không thể tải dữ liệu cho các mã cổ phiếu đã chọn!")
            st.stop()
        
        st.success(f"Đã tải dữ liệu cho {len(prices.columns)} mã cổ phiếu từ {start_date} đến {end_date}")
        
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {str(e)}")
        st.stop()
else:
    st.warning("Vui lòng nhập ít nhất một mã cổ phiếu!")
    st.stop()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Thống kê Cổ phiếu", "📈 Phân tích Danh mục", "⚖️ CAPM & SML", "🔮 Dự báo LSTM"])

# Tab 1: Stock Profile
with tab1:
    st.subheader("Thống kê và Phân tích Cổ phiếu")
    
    # Price chart
    st.subheader("Biểu đồ Giá")
    fig_price = go.Figure()
    
    for symbol in prices.columns:
        fig_price.add_trace(go.Scatter(
            x=prices.index,
            y=prices[symbol],
            mode='lines',
            name=symbol,
            hovertemplate=f'{symbol}: %{{y:,.0f}} VND<br>Ngày: %{{x}}<extra></extra>'
        ))
    
    fig_price.update_layout(
        title="Diễn biến Giá Cổ phiếu",
        xaxis_title="Thời gian",
        yaxis_title="Giá (VND)",
        hovermode='x unified',
        showlegend=False,
        height=500
    )
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Statistics table
    summary = summarize_all_stocks(returns, rf_annual)
    
    if not summary.empty:
        st.subheader("Thống kê Tổng quan")
        
        # Hiển thị bảng
        st.dataframe(
            summary,
            column_config={
                "Lợi nhuận TB (% năm)": st.column_config.NumberColumn(
                    "Lợi nhuận TB (% năm)", format="%.2f"),
                "Độ lệch chuẩn (% năm)": st.column_config.NumberColumn(
                    "Độ lệch chuẩn (% năm)", format="%.2f"),
                "Sharpe Ratio": st.column_config.NumberColumn(
                    "Sharpe Ratio", format="%.3f"),
                "VaR 5% (% ngày)": st.column_config.NumberColumn(
                    "VaR 5% (% ngày)", format="%.2f"),
                "Max Drawdown (%)": st.column_config.NumberColumn(
                    "Max Drawdown (%)", format="%.2f")
            }
        )
    else:
        st.warning("Không thể tính thống kê cho dữ liệu này")
    
    # Explanations  
    with st.expander("📖 Giải thích các chỉ số"):
        st.markdown("""
        - **Lợi nhuận TB**: Tỷ suất lợi nhuận trung bình năm hóa (%)
        - **Độ lệch chuẩn**: Mức độ biến động rủi ro năm hóa (%)
        - **Sharpe Ratio**: Tỷ số lợi nhuận/rủi ro (càng cao càng tốt)
        - **VaR 5%**: Tổn thất tối đa có thể xảy ra trong 5% trường hợp xấu nhất
        - **Max Drawdown**: Mức giảm tối đa từ đỉnh đến đáy (%)
        - **Skewness**: Độ lệch phân phối (>0: lệch phải, <0: lệch trái) 
        - **Kurtosis**: Độ nhọn phân phối (>0: nhọn hơn chuẩn)
        """)

# Tab 2: Portfolio Analysis  
with tab2:
    st.subheader("Phân tích Danh mục Đầu tư")
    
    if not returns.empty:
        # Correlation Matrix
        st.subheader("Ma trận Tương quan")
        correlation_matrix = returns.corr()
        
        st.write("**Ma trận Tương quan:**")
        st.dataframe(
            correlation_matrix,
            column_config={col: st.column_config.NumberColumn(col, format="%.3f") 
                          for col in correlation_matrix.columns}
        )
        
        # Covariance Matrix
        st.subheader("Ma trận Hiệp phương sai (Annualized)")
        covariance_matrix = returns.cov() * 252  # Annualized
        
        st.write("**Ma trận Hiệp phương sai (Năm hóa):**")
        st.dataframe(
            covariance_matrix,
            column_config={col: st.column_config.NumberColumn(col, format="%.6f") 
                          for col in covariance_matrix.columns}
        )
        
        # Risk-Return Analysis  
        st.subheader("Phân tích Rủi ro - Lợi nhuận")
        
        risk_return_data = []
        for stock in returns.columns:
            stock_returns = returns[stock].dropna()
            if len(stock_returns) > 20:
                annual_return = stock_returns.mean() * 252 * 100  # %
                annual_risk = stock_returns.std() * np.sqrt(252) * 100  # %
                sharpe = (annual_return - rf_annual * 100) / annual_risk if annual_risk > 0 else 0
                
                risk_return_data.append({
                    'Cổ phiếu': stock,
                    'Lợi nhuận (%)': annual_return,
                    'Rủi ro (%)': annual_risk,
                    'Sharpe Ratio': sharpe
                })
        
        if risk_return_data:
            df_risk_return = pd.DataFrame(risk_return_data)
            
            # Display risk-return table
            st.dataframe(
                df_risk_return,
                column_config={
                    "Lợi nhuận (%)": st.column_config.NumberColumn("Lợi nhuận (%)", format="%.2f"),
                    "Rủi ro (%)": st.column_config.NumberColumn("Rủi ro (%)", format="%.2f"),
                    "Sharpe Ratio": st.column_config.NumberColumn("Sharpe Ratio", format="%.3f")
                }
            )
            
            # Risk-Return scatter plot
            fig_scatter = go.Figure()
            
            fig_scatter.add_trace(go.Scatter(
                x=df_risk_return['Rủi ro (%)'],
                y=df_risk_return['Lợi nhuận (%)'],
                mode='markers+text',
                text=df_risk_return['Cổ phiếu'],
                textposition="top center",
                marker=dict(
                    size=15,
                    color=df_risk_return['Sharpe Ratio'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio")
                ),
                name='Cổ phiếu'
            ))
            
            fig_scatter.update_layout(
                title='Biểu đồ Rủi ro - Lợi nhuận',
                xaxis_title='Rủi ro (% năm)',
                yaxis_title='Lợi nhuận (% năm)',
                showlegend=False,
                height=500
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Summary statistics
            st.subheader("📋 Thống kê Tóm tắt Danh mục")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Lợi nhuận TB", f"{df_risk_return['Lợi nhuận (%)'].mean():.2f}%")
                st.metric("Rủi ro TB", f"{df_risk_return['Rủi ro (%)'].mean():.2f}%")
            
            with col2:
                st.metric("Sharpe Ratio TB", f"{df_risk_return['Sharpe Ratio'].mean():.3f}")
                avg_corr = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
                st.metric("Tương quan TB", f"{avg_corr:.3f}")
            
            with col3:
                best_idx = df_risk_return['Sharpe Ratio'].idxmax()
                worst_idx = df_risk_return['Sharpe Ratio'].idxmin()
                st.metric("Sharpe tốt nhất", df_risk_return.loc[best_idx, 'Cổ phiếu'])
                st.metric("Sharpe kém nhất", df_risk_return.loc[worst_idx, 'Cổ phiếu'])
            
            # Efficient Frontier Analysis
            st.subheader("Đường Biên Hiệu Quả & Tối Ưu Hóa Danh Mục")
            
            # User input for expected returns
            st.write("**Nhập tỷ suất lợi nhuận kỳ vọng cho từng cổ phiếu:**")
            
            expected_returns_input = []
            col_inputs = st.columns(min(3, len(symbols)))
            
            for i, stock in enumerate(symbols):
                with col_inputs[i % 3]:
                    # Use historical return as default
                    default_return = df_risk_return[df_risk_return['Cổ phiếu'] == stock]['Lợi nhuận (%)'].iloc[0] / 100
                    expected_ret = st.number_input(
                        f"E[R] {stock} (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(default_return) * 100,
                        step=1.0,
                        key=f"er_{stock}"
                    ) / 100
                    expected_returns_input.append(expected_ret)
            
            # Portfolio type selection
            portfolio_type = st.selectbox(
                "Chọn loại danh mục tối ưu:",
                ["min_var", "max_sharpe"],
                format_func=lambda x: "Rủi ro thấp nhất (Min Variance)" if x == "min_var" else "Sharpe cao nhất (Max Sharpe)"
            )
            
            # Investment amount input
            investment_amount = st.number_input(
                "Số tiền đầu tư (VND)",
                min_value=1000000,
                max_value=10000000000,
                value=10000000,
                step=1000000,
                format="%d"
            )
            
            if st.button("Tính toán Đường Biên Hiệu Quả", type="primary"):
                try:
                    from src.efficient_frontier import EfficientFrontierAnalyzer
                    
                    if not prices.empty:
                        # Create analyzer with expected returns
                        ef_analyzer = EfficientFrontierAnalyzer(prices, expected_returns_input)
                        
                        # Find both optimal portfolios
                        min_var_portfolio = ef_analyzer.find_minimum_variance_portfolio()
                        max_sharpe_portfolio = ef_analyzer.find_max_sharpe_portfolio()
                        
                        # Build efficient frontier
                        ef_analyzer.build_efficient_frontier()
                        
                        # Display summary metrics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Danh mục Phương sai Tối thiểu")
                            st.metric("Lợi nhuận kỳ vọng", f"{min_var_portfolio['return']*100:.2f}%/năm")
                            st.metric("Rủi ro", f"{min_var_portfolio['volatility']*100:.2f}%/năm")
                            st.metric("Sharpe Ratio", f"{min_var_portfolio['sharpe']:.3f}")
                        
                        with col2:
                            st.subheader("Danh mục Sharpe Tối đa")
                            st.metric("Lợi nhuận kỳ vọng", f"{max_sharpe_portfolio['return']*100:.2f}%/năm")
                            st.metric("Rủi ro", f"{max_sharpe_portfolio['volatility']*100:.2f}%/năm")
                            st.metric("Sharpe Ratio", f"{max_sharpe_portfolio['sharpe']:.3f}")
                        
                        # Plot efficient frontier
                        st.subheader("Biểu đồ Đường Biên Hiệu Quả")
                        fig_ef = ef_analyzer.plot_efficient_frontier()
                        st.plotly_chart(fig_ef, use_container_width=True)
                        
                        # Investment recommendation for selected portfolio type
                        st.subheader("Khuyến nghị Đầu tư")
                        recommendation_df, selected_portfolio, portfolio_name = ef_analyzer.get_investment_recommendation(
                            investment_amount, portfolio_type
                        )
                        
                        st.write(f"**Danh mục được chọn:** {portfolio_name}")
                        st.dataframe(
                            recommendation_df,
                            column_config={
                                "Tỷ trọng (%)": st.column_config.NumberColumn("Tỷ trọng (%)", format="%.2f"),
                                "Số tiền đầu tư (VND)": st.column_config.NumberColumn("Số tiền (VND)", format="%.0f"),
                                "Số tiền đầu tư (triệu VND)": st.column_config.NumberColumn("Số tiền (triệu)", format="%.2f")
                            },
                            use_container_width=True
                        )
                        
                        # Display efficient frontier table
                        with st.expander("Bảng Chi tiết Đường Biên Hiệu Quả"):
                            ef_df = ef_analyzer.get_portfolio_allocations_df()
                            
                            st.write("**Top 20 danh mục hiệu quả:**")
                            display_df = ef_df.head(20)
                            
                            column_config = {
                                "Portfolio": st.column_config.NumberColumn("STT", format="%d"),
                                "Risk (Volatility)": st.column_config.NumberColumn("Rủi ro", format="%.4f"),
                                "Return": st.column_config.NumberColumn("Lợi nhuận", format="%.4f"),
                                "Sharpe Ratio": st.column_config.NumberColumn("Sharpe", format="%.3f")
                            }
                            for stock in symbols:
                                column_config[f"{stock} (%)"] = st.column_config.NumberColumn(f"{stock} (%)", format="%.1f")
                            
                            st.dataframe(
                                display_df,
                                column_config=column_config,
                                use_container_width=True
                            )
                        
                        # Investment advice
                        st.subheader("Lời khuyên Đầu tư")
                        
                        # Get selected portfolio info
                        if portfolio_type == "min_var":
                            selected_portfolio = min_var_portfolio
                            selected_name = "Min Variance"
                        else:
                            selected_portfolio = max_sharpe_portfolio
                            selected_name = "Max Sharpe"
                        
                        # Build investment advice as clean Markdown (avoid triple-quoted indentation)
                        advice_lines = []
                        advice_lines.append("Dựa trên phân tích đường biên hiệu quả Markowitz, chúng tôi đưa ra 2 lựa chọn danh mục tối ưu:")
                        advice_lines.append("")
                        advice_lines.append(f"**Danh mục Min Variance** (Rủi ro thấp nhất): Phù hợp với nhà đầu tư ưa thích an toàn, mong muốn bảo toàn vốn với lợi nhuệm ổn định {min_var_portfolio['return']*100:.1f}%/năm và rủi ro chỉ {min_var_portfolio['volatility']*100:.1f}%/năm.")
                        advice_lines.append("")
                        advice_lines.append(f"**Danh mục Max Sharpe** (Hiệu quả cao nhất): Phù hợp với nhà đầu tư muốn tối ưu hóa tỷ lệ lợi nhuận/rủi ro với Sharpe ratio {max_sharpe_portfolio['sharpe']:.2f}, mang lại lợi nhuận {max_sharpe_portfolio['return']*100:.1f}%/năm.")
                        advice_lines.append("")
                        advice_lines.append(f"Với số vốn {investment_amount/1000000:.0f} triệu VND, bạn nên phân bổ theo danh mục **{selected_name}** như sau:")
                        advice_lines.append("")

                        # Add individual stock allocations as list items
                        for _, row in recommendation_df.iterrows():
                            advice_lines.append(f"- **{row['Mã CP']}**: {row['Số tiền đầu tư (triệu VND)']:.1f} triệu VND ({row['Tỷ trọng (%)']:.1f}%)")

                        advice_lines.append("")
                        advice_lines.append("Lưu ý: Đây là phân tích dựa trên dữ liệu lịch sử, thị trường thực tế có thể biến động khác biệt.")

                        advice_text = "\n".join(advice_lines)

                        st.markdown(advice_text)
                        
                        if portfolio_type == "min_var":
                            st.info("Danh mục Min Variance phù hợp với nhà đầu tư ưa thích an toàn, tối thiểu rủi ro.")
                        else:
                            st.info("Danh mục Max Sharpe phù hợp với nhà đầu tư muốn tối ưu hóa tỷ lệ lợi nhuận/rủi ro.")
                        
                        st.warning("Lưu ý: Đây là mô hình lý thuyết dựa trên dữ liệu lịch sử. Thị trường thực tế có thể biến động khác biệt.")
                        
                    else:
                        st.error("Không có dữ liệu giá để tính toán đường biên hiệu quả")
                        
                except Exception as e:
                    st.error(f"Lỗi khi tính toán đường biên hiệu quả: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    else:
        st.warning("Không có dữ liệu để phân tích danh mục")

# Tab 3: CAPM Analysis
with tab3:
    st.subheader("CAPM & Đường Thị trường Chứng khoán (SML)")
    
    # Calculate CAPM metrics
    # Tính actual returns trực tiếp từ returns data
    E_R_actual = {}
    for stock in symbols:
        if stock in returns.columns:
            stock_returns = returns[stock].dropna()
            if len(stock_returns) > 20:
                annual_return = stock_returns.mean() * 252  # Annualized
                E_R_actual[stock] = annual_return
    
    mean_daily_mkt = mkt_returns.mean()
    E_Rm_annual = mean_daily_mkt * 252
    
    capm_results = []
    betas = {}
    
    # Initialize CAPM analyzer once with all stocks
    try:
        from src.capm import CAPMAnalyzer
        capm_analyzer = CAPMAnalyzer(returns, mkt_returns, rf_annual)
        
        for stock in symbols:
            if stock in returns.columns:
                try:
                    beta_stats = capm_analyzer.calculate_beta(stock)
                    
                    if 'error' not in beta_stats:
                        betas[stock] = beta_stats['beta']
                        expected_return_capm = rf_annual + beta_stats['beta'] * (E_Rm_annual - rf_annual)
                        actual_return = E_R_actual.get(stock, 0)
                        
                        capm_results.append({
                            'Mã CP': stock,
                            'Beta': beta_stats['beta'],
                            'E[R] thực tế (%)': actual_return * 100,
                            'E[R] CAPM (%)': expected_return_capm * 100,
                            'Alpha (%)': (actual_return - expected_return_capm) * 100,
                            'R²': beta_stats['r_squared']
                        })
                    else:
                        st.warning(f"Không thể tính Beta cho {stock}: {beta_stats['error']}")
                        
                except Exception as e:
                    st.warning(f"Lỗi CAPM cho {stock}: {str(e)}")
    
    except Exception as e:
        st.error(f"Lỗi khởi tạo CAPM: {str(e)}")
    
    if capm_results:
        # Display CAPM results
        capm_df = pd.DataFrame(capm_results)
        st.dataframe(
            capm_df,
            column_config={
                "Beta": st.column_config.NumberColumn("Beta", format="%.3f"),
                "E[R] thực tế (%)": st.column_config.NumberColumn("E[R] thực tế (%)", format="%.2f"),
                "E[R] CAPM (%)": st.column_config.NumberColumn("E[R] CAPM (%)", format="%.2f"),
                "Alpha (%)": st.column_config.NumberColumn("Alpha (%)", format="%.2f"),
                "R²": st.column_config.NumberColumn("R²", format="%.3f")
            }
        )
        
        # Plot SML with error handling
        try:
            from src.capm import CAPMAnalyzer
            capm_analyzer = CAPMAnalyzer(returns, mkt_returns, rf_annual)
            fig_sml = capm_analyzer.plot_security_market_line()
            
            if fig_sml:
                st.plotly_chart(fig_sml, use_container_width=True)
            else:
                st.warning("Không thể vẽ biểu đồ SML")
                
        except Exception as e:
            st.error(f"Lỗi khi vẽ SML: {str(e)}")
    
    else:
        st.warning("Không thể tính CAPM cho các cổ phiếu này")

# Tab 4: LSTM Forecast
with tab4:
    st.subheader(" Dự báo Giá sử dụng LSTM")
    
    # Select stock for forecast
    forecast_stock = st.selectbox("Chọn cổ phiếu để dự báo:", symbols)
    
    if forecast_stock and forecast_stock in prices.columns:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            lookback_years = st.slider("Số năm nhìn lại:", 1, 3, 1)
            lookback_days = int(lookback_years * 252)  # Convert years to trading days
        with col2:
            forecast_days = st.slider("Số ngày dự báo:", 7, 60, 30)
        
        if st.button("Chạy Dự báo LSTM"):
            try:
                from src.lstm_forecast import run_lstm_analysis, simple_moving_average_forecast
                
                stock_prices = prices[forecast_stock].dropna()
                
                if len(stock_prices) < lookback_days + 50:
                    st.error(f"Không đủ dữ liệu cho {forecast_stock}. Cần {lookback_days + 50} ngày, có {len(stock_prices)} ngày.")
                else:
                    with st.spinner("Đang phân tích..."):
                        lstm_results = run_lstm_analysis(stock_prices, forecast_stock, lookback_days, forecast_days)
                        
                        if lstm_results['success']:
                            # Metrics
                            metrics = lstm_results['training']['metrics']
                            forecast_results = lstm_results['forecast']
                            current_price = stock_prices.iloc[-1]
                            predicted_price = forecast_results['predictions'][-1]
                            pred_change = (predicted_price / current_price - 1) * 100
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("RMSE", f"{metrics['test_rmse']:.0f}")
                            with col2:
                                st.metric("Giá hiện tại", f"{current_price:,.0f} VND")
                            with col3:
                                st.metric(f"Dự báo {forecast_days}d", f"{predicted_price:,.0f} VND", f"{pred_change:+.1f}%")
                            
                            # Create forecast plot với enhanced red line
                            fig_forecast = go.Figure()
                            
                            # Historical prices (recent period for better visualization)
                            display_days = min(lookback_days // 2, len(stock_prices), 100)
                            recent_prices = stock_prices.tail(display_days)
                            
                            # Add historical prices (BLUE)
                            fig_forecast.add_trace(go.Scatter(
                                x=recent_prices.index,
                                y=recent_prices.values,
                                mode='lines',
                                name='Lịch sử',
                                line=dict(color='blue', width=2),
                                hovertemplate='<b>Lịch sử</b><br>Ngày: %{x}<br>Giá: %{y:,.0f} VND<extra></extra>',
                                showlegend=True
                            ))
                            
                            # Extract LSTM forecast data
                            if 'predictions' in forecast_results and len(forecast_results['predictions']) > 0:
                                forecast_predictions = np.array(forecast_results['predictions']).flatten()
                                
                                # Get or create forecast dates with robust handling
                                if 'dates' in forecast_results and len(forecast_results['dates']) == len(forecast_predictions):
                                    forecast_dates = forecast_results['dates']
                                else:
                                    # Create forecast dates manually with robust date handling
                                    last_date = recent_prices.index[-1]
                                    
                                    try:
                                        if isinstance(last_date, (int, np.integer)):
                                            last_date = pd.Timestamp.today()
                                        elif isinstance(last_date, str):
                                            last_date = pd.to_datetime(last_date)
                                        elif hasattr(last_date, 'date') or isinstance(last_date, pd.Timestamp):
                                            last_date = pd.to_datetime(last_date)
                                        else:
                                            last_date = pd.Timestamp.today()
                                    except:
                                        last_date = pd.Timestamp.today()
                                    
                                    forecast_dates = pd.date_range(
                                        start=last_date + pd.Timedelta(days=1),
                                        periods=len(forecast_predictions),
                                        freq='D'
                                    )
                                
                                # Connection line from last historical to first forecast (ORANGE)
                                fig_forecast.add_trace(go.Scatter(
                                    x=[recent_prices.index[-1], forecast_dates[0]],
                                    y=[recent_prices.iloc[-1], forecast_predictions[0]],
                                    mode='lines',
                                    line=dict(color='orange', width=2, dash='dot'),
                                    name='Kết nối',
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                                
                                # 🔴 LSTM FORECAST LINE - ENHANCED FOR VISIBILITY
                                fig_forecast.add_trace(go.Scatter(
                                    x=forecast_dates,
                                    y=forecast_predictions,
                                    mode='lines+markers',
                                    name='🔴 Dự báo LSTM',
                                    line=dict(
                                        color='red', 
                                        width=4,  # Thicker line
                                        dash=None  # Solid line
                                    ),
                                    marker=dict(
                                        size=8, 
                                        color='red',
                                        symbol='circle',
                                        line=dict(width=2, color='darkred')
                                    ),
                                    visible=True,  # Force visible
                                    opacity=1.0,   # Full opacity
                                    hovertemplate='<b>🔴 Dự báo LSTM</b><br>Ngày: %{x}<br>Giá: %{y:,.0f} VND<extra></extra>',
                                    showlegend=True
                                ))
                                
                                # Confidence interval (light red shading)
                                ci = forecast_results.get('confidence_interval', {})
                                if 'upper' in ci and 'lower' in ci and len(ci['upper']) == len(forecast_dates):
                                    # Upper bound (invisible line)
                                    fig_forecast.add_trace(go.Scatter(
                                        x=forecast_dates,
                                        y=ci['upper'],
                                        fill=None,
                                        mode='lines',
                                        line_color='rgba(0,0,0,0)',
                                        showlegend=False,
                                        hoverinfo='skip'
                                    ))
                                    
                                    # Lower bound with fill
                                    fig_forecast.add_trace(go.Scatter(
                                        x=forecast_dates,
                                        y=ci['lower'],
                                        fill='tonexty',
                                        mode='lines',
                                        line_color='rgba(0,0,0,0)',
                                        fillcolor='rgba(255,0,0,0.15)',  # Light red fill
                                        name='Khoảng tin cậy',
                                        hovertemplate='Khoảng tin cậy<extra></extra>'
                                    ))
                            
                            # Chart styling với enhanced layout
                            fig_forecast.update_layout(
                                title={
                                    'text': f'🔮 Dự báo LSTM {forecast_days} ngày - {forecast_stock}',
                                    'x': 0.5,
                                    'font': {'size': 16}
                                },
                                xaxis_title='Ngày',
                                yaxis_title='Giá (VND)',
                                height=600,
                                hovermode='x unified',
                                showlegend=True,
                                legend=dict(
                                    x=0.01, y=0.99,
                                    bgcolor='rgba(255,255,255,0.8)',
                                    bordercolor='gray',
                                    borderwidth=1
                                ),
                                template='plotly_white'
                            )
                            
                            # Add forecast summary annotation
                            current_price = recent_prices.iloc[-1]
                            final_forecast = forecast_predictions[-1] if len(forecast_predictions) > 0 else current_price
                            change_pct = (final_forecast / current_price - 1) * 100
                            
                            fig_forecast.add_annotation(
                                x=0.02, y=0.98,
                                xref="paper", yref="paper",
                                text=f"<b>Tóm tắt Dự báo</b><br>" +
                                     f"Hiện tại: {current_price:,.0f} VND<br>" +
                                     f"{forecast_days} ngày: {final_forecast:,.0f} VND<br>" +
                                     f"Thay đổi: {change_pct:+.1f}%",
                                showarrow=False,
                                font=dict(size=11, color="black"),
                                bgcolor="rgba(255,255,255,0.8)",
                                bordercolor="gray",
                                borderwidth=1
                            )
                            
                            st.plotly_chart(fig_forecast, use_container_width=True)
                            
                            # Success message
                            st.success("✅ Biểu đồ LSTM với đường dự báo màu đỏ đã được hiển thị!")
                            
                            # Forecast interpretation
                            if pred_change > 5:
                                trend = "🚀 Xu hướng tăng mạnh"
                                st.success(trend)
                            elif pred_change > 0:
                                trend = "📈 Xu hướng tăng nhẹ"
                                st.info(trend)
                            elif pred_change > -5:
                                trend = "📉 Xu hướng giảm nhẹ"
                                st.warning(trend)
                            else:
                                trend = "📉 Xu hướng giảm mạnh"
                                st.error(trend)
                        
                        else:
                            st.warning(f"LSTM không thành công: {lstm_results.get('error', 'Lỗi không xác định')}")
                            st.info("Chuyển sang Moving Average...")
                            
                            ma_results = simple_moving_average_forecast(stock_prices, forecast_days)
                            
                            if ma_results['success']:
                                current_price = stock_prices.iloc[-1]
                                predicted_price = ma_results['predictions'][-1]
                                pred_change = (predicted_price / current_price - 1) * 100
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Phương pháp", "Moving Average")
                                with col2:
                                    st.metric("Giá hiện tại", f"{current_price:,.0f} VND")
                                with col3:
                                    st.metric(f"Dự báo {forecast_days}d", f"{predicted_price:,.0f} VND", f"{pred_change:+.1f}%")
                                
                                # Enhanced MA plot
                                fig_ma = go.Figure()
                                recent_prices = stock_prices.tail(100)
                                
                                # Historical data (blue)
                                fig_ma.add_trace(go.Scatter(
                                    x=recent_prices.index, 
                                    y=recent_prices.values, 
                                    name='Lịch sử', 
                                    line=dict(color='blue', width=2),
                                    hovertemplate='<b>Lịch sử</b><br>Ngày: %{x}<br>Giá: %{y:,.0f} VND<extra></extra>'
                                ))
                                
                                # MA forecast (red)
                                fig_ma.add_trace(go.Scatter(
                                    x=ma_results['dates'], 
                                    y=ma_results['predictions'], 
                                    name='🔴 Dự báo MA', 
                                    line=dict(color='red', width=4),
                                    mode='lines+markers',
                                    marker=dict(size=6, color='red'),
                                    hovertemplate='<b>🔴 Dự báo MA</b><br>Ngày: %{x}<br>Giá: %{y:,.0f} VND<extra></extra>'
                                ))
                                
                                # Confidence interval
                                ci = ma_results['confidence_interval']
                                fig_ma.add_trace(go.Scatter(x=ma_results['dates'], y=ci['upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
                                fig_ma.add_trace(go.Scatter(x=ma_results['dates'], y=ci['lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', fillcolor='rgba(255,0,0,0.15)', name='Khoảng tin cậy'))
                                
                                fig_ma.update_layout(
                                    title=f'📊 Dự báo Moving Average - {forecast_stock}', 
                                    xaxis_title='Ngày', 
                                    yaxis_title='Giá (VND)', 
                                    showlegend=True, 
                                    height=500,
                                    template='plotly_white'
                                )
                                st.plotly_chart(fig_ma, use_container_width=True)
                            else:
                                st.error("Không thể dự báo")
                
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")
    else:
        st.warning("Chọn cổ phiếu để dự báo")

# Footer
st.markdown("---")
st.markdown("*Khai phá web | Ứng dụng hỗ trợ ra quyết định đầu tư cổ phiếu*")