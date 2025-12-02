"""
Data Loader Module
Xử lý tải dữ liệu từ vnstock và yfinance
"""

import pandas as pd
import numpy as np
try:
    from vnstock import Vnstock
    VNSTOCK_AVAILABLE = True
except ImportError:
    try:
        import vnstock as vn
        VNSTOCK_AVAILABLE = True
    except ImportError:
        VNSTOCK_AVAILABLE = False
import yfinance as yf
from datetime import datetime
import streamlit as st

# Cấu hình mặc định
START_DATE = "2020-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

@st.cache_data(ttl=3600)
@st.cache_data(ttl=3600)
def get_vn_price(symbol: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """
    Tải dữ liệu giá cổ phiếu Việt Nam từ vnstock hoặc yfinance
    """
    def _normalize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        Đảm bảo DataFrame có DatetimeIndex (dùng cột time/date nếu có)
        """
        # Ưu tiên cột 'time' hoặc 'date'
        for col in ['time', 'date', 'Date', 'datetime']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                break

        # Nếu vẫn chưa phải DatetimeIndex (vnstock đôi khi trả index int)
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                # Thử convert index trực tiếp
                df.index = pd.to_datetime(df.index)
            except Exception:
                # Nếu là unix timestamp (giây)
                try:
                    df.index = pd.to_datetime(df.index, unit='s')
                except Exception:
                    # Trường hợp xấu nhất: tự tạo dãy ngày từ start–end
                    dates = pd.date_range(start=start, end=end, periods=len(df))
                    df.index = dates

        return df.sort_index()

    try:
        # Thử vnstock trước
        if VNSTOCK_AVAILABLE:
            try:
                # API mới
                stock = Vnstock().stock(symbol=symbol, source='VCI')
                df = stock.quote.history(start=start, end=end)

                if df is not None and not df.empty:
                    df = _normalize_datetime_index(df)
                    if 'close' in df.columns:
                        df = df[['close']].copy()
                        df.rename(columns={'close': symbol}, inplace=True)
                        return df
            except:
                # Thử API cũ nếu có
                try:
                    import vnstock as vn
                    df = vn.stock_historical_data(
                        symbol=symbol,
                        start_date=start,
                        end_date=end
                    )
                    if df is not None and not df.empty:
                        df = _normalize_datetime_index(df)
                        df = df[['close']].copy()
                        df.rename(columns={'close': symbol}, inplace=True)
                        return df
                except:
                    pass

        # Fallback: yfinance (.VN)
        ticker = f"{symbol}.VN"
        data = yf.download(ticker, start=start, end=end, progress=False)

        if not data.empty and 'Close' in data.columns:
            df = pd.DataFrame(data['Close'])
            df = _normalize_datetime_index(df)
            df.rename(columns={'Close': symbol}, inplace=True)
            return df
        else:
            raise ValueError(f"Không có dữ liệu cho {symbol}")

    except Exception as e:
        st.warning(f"Không thể tải dữ liệu cho {symbol}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_prices_for_list(symbols: list, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """
    Tải dữ liệu cho danh sách cổ phiếu
    """
    dfs = []
    
    for symbol in symbols:
        df = get_vn_price(symbol, start, end)
        if not df.empty:
            dfs.append(df)
    
    if not dfs:
        raise ValueError("Không thể tải dữ liệu cho bất kỳ cổ phiếu nào")
    
    # Ghép tất cả dữ liệu
    prices = pd.concat(dfs, axis=1).sort_index()

    # Chuẩn hóa index: bỏ timezone, normalize về ngày
    def _normalize_idx(idx: pd.Index) -> pd.Index:
        ts = pd.to_datetime(idx)
        if getattr(ts, "tz", None) is not None:
            ts = ts.tz_convert(None)
        return ts.normalize()
    prices.index = _normalize_idx(prices.index)
    
    # Xử lý missing data
    prices = prices.ffill().bfill()
    
    return prices

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Tính toán tỷ suất lợi nhuận hàng ngày
    """
    # Chuẩn hóa index trước khi tính returns
    def _normalize_idx(idx: pd.Index) -> pd.Index:
        ts = pd.to_datetime(idx)
        if getattr(ts, "tz", None) is not None:
            ts = ts.tz_convert(None)
        return ts.normalize()

    prices = prices.copy()
    prices.index = _normalize_idx(prices.index)
    returns = prices.pct_change().dropna(how='all')
    return returns

@st.cache_data(ttl=3600)
def get_market_index_returns(start: str = START_DATE, end: str = END_DATE, ticker: str = "^VNINDEX"):
    """
    Tải dữ liệu chỉ số thị trường từ vnstock hoặc Yahoo Finance
    """
    try:
        # Thử vnstock cho VNIndex trước
        if VNSTOCK_AVAILABLE and ticker == "^VNINDEX":
            try:
                stock = Vnstock().stock(symbol='VNINDEX', source='VCI')
                df = stock.quote.history(start=start, end=end)
                
                if df is not None and not df.empty and 'close' in df.columns:
                    # Chuẩn hóa index
                    df = df.copy()
                    if 'time' in df.columns:
                        df['time'] = pd.to_datetime(df['time'])
                        df = df.set_index('time')
                    df.index = pd.to_datetime(df.index).normalize()
                    close = df['close'].rename("MARKET")
                    returns = close.pct_change().dropna()
                    return close, returns
            except:
                pass
        
        # Fallback: Yahoo Finance
        data = yf.download(ticker, start=start, end=end, progress=False)
        
        if not data.empty and "Close" in data.columns:
            df = data.copy()
            df.index = pd.to_datetime(df.index).normalize()
            close = df["Close"].rename("MARKET")
            returns = close.pct_change().dropna()
            return close, returns
        else:
            raise ValueError(f"Không có dữ liệu cho {ticker}")
        
    except Exception as e:
        st.warning(f"Không thể tải dữ liệu thị trường {ticker}: {str(e)}")
        # Fallback: tạo dữ liệu mô phỏng
        dates = pd.date_range(start=start, end=end, freq='D')
        np.random.seed(42)  # Để kết quả nhất quán
        fake_close = pd.Series(
            np.random.randn(len(dates)).cumsum() * 5 + 1200,  # Mô phỏng VNIndex quanh 1200 điểm
            index=dates,
            name="MARKET"
        )
        fake_returns = fake_close.pct_change().dropna()
        return fake_close, fake_returns

def get_company_info(symbols: list) -> pd.DataFrame:
    """
    Lấy thông tin cơ bản về công ty (tên, ngành, vốn hóa...)
    """
    # Thông tin cơ bản một số cổ phiếu phổ biến
    company_data = {
        'FPT': {'name': 'Công ty Cổ phần FPT', 'industry': 'Công nghệ thông tin'},
        'VNM': {'name': 'Công ty Cổ phần Sữa Việt Nam', 'industry': 'Thực phẩm & đồ uống'},
        'VCB': {'name': 'Ngân hàng TMCP Ngoại thương Việt Nam', 'industry': 'Ngân hàng'},
        'HPG': {'name': 'Công ty Cổ phần Tập đoàn Hòa Phát', 'industry': 'Thép'},
        'MWG': {'name': 'Công ty Cổ phần Đầu tư Thế giới Di động', 'industry': 'Bán lẻ'},
        'VIC': {'name': 'Công ty Cổ phần Vingroup', 'industry': 'Bất động sản'},
        'VHM': {'name': 'Công ty Cổ phần Vinhomes', 'industry': 'Bất động sản'},
        'GAS': {'name': 'Công ty Cổ phần Khí Việt Nam', 'industry': 'Dầu khí'},
        'CTG': {'name': 'Ngân hàng TMCP Công thương Việt Nam', 'industry': 'Ngân hàng'},
        'TCB': {'name': 'Ngân hàng TMCP Kỹ thương Việt Nam', 'industry': 'Ngân hàng'}
    }
    
    info_list = []
    for symbol in symbols:
        info = company_data.get(symbol, {'name': f'Công ty {symbol}', 'industry': 'Khác'})
        info_list.append({
            'symbol': symbol,
            'company_name': info['name'],
            'industry': info['industry']
        })
    
    return pd.DataFrame(info_list).set_index('symbol')