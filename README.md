# Ứng dụng hỗ trợ nhà đầu tư cổ phiếu Việt Nam

Ứng dụng phân tích và tối ưu hóa danh mục đầu tư dựa trên:
- Lý thuyết Markowitz
- Mô hình CAPM
- Dự báo LSTM

## Cài đặt

```bash
pip install streamlit pandas numpy yfinance vnstock plotly scikit-learn scipy tensorflow
```

## Chạy ứng dụng

```bash
streamlit run app.py
```

## Tính năng

1. **Stock Profile**: Phân tích thống kê cơ bản cổ phiếu
2. **Portfolio Markowitz**: Tối ưu hóa danh mục theo lý thuyết hiện đại
3. **CAPM & SML**: Phân tích rủi ro và lợi nhuận theo CAPM
4. **LSTM Forecast**: Dự báo giá sử dụng Deep Learning

## Cấu trúc project

```
├── app.py              # Ứng dụng Streamlit chính
├── src/
│   ├── data_loader.py      # Module tải dữ liệu
│   ├── analysis_basic.py   # Phân tích cơ bản
│   ├── portfolio_markowitz.py  # Tối ưu hóa Markowitz
│   ├── capm.py            # Phân tích CAPM
│   └── lstm_forecast.py   # Dự báo LSTM
```

## Sử dụng

1. Chọn danh sách cổ phiếu Việt Nam
2. Thiết lập khoảng thời gian
3. Phân tích qua 4 tab chính
4. Tải xuống kết quả

## Lưu ý

- Sử dụng mã cổ phiếu Việt Nam (VD: VNM, FPT, VCB)
- Cần kết nối internet để tải dữ liệu
- TensorFlow cần thiết cho tính năng LSTM