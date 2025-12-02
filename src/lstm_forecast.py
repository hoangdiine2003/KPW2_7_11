"""
LSTM Forecast Module - TensorFlow Implementation
Dự báo giá cổ phiếu sử dụng mạng nơ-ron LSTM với TensorFlow
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
    
    # Configure TensorFlow to avoid warnings
    tf.get_logger().setLevel('ERROR')
    
except ImportError:
    TENSORFLOW_AVAILABLE = False

class LSTMPredictor:
    """LSTM Neural Network với TensorFlow"""
    
    def __init__(self, lookback_days: int = 60, forecast_days: int = 30):
        self.lookback_days = lookback_days
        self.forecast_days = forecast_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.is_trained = False
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow không được cài đặt. Vui lòng cài đặt: pip install tensorflow")
    
    def prepare_data(self, prices: pd.Series):
        """Chuẩn bị dữ liệu cho LSTM"""
        # Scale dữ liệu
        data = prices.values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.lookback_days, len(scaled_data)):
            X.append(scaled_data[i-self.lookback_days:i, 0])
            y.append(scaled_data[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape cho LSTM (samples, time steps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def build_model(self):
        """Xây dựng mô hình LSTM"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.lookback_days, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, prices: pd.Series, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2) -> dict:
        """Huấn luyện mô hình LSTM"""
        try:
            X, y = self.prepare_data(prices)
            
            if len(X) < 100:
                return {'success': False, 'error': 'Không đủ dữ liệu cho LSTM (cần ít nhất 100 mẫu)'}
            
            # Chia dữ liệu train/test
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Xây dựng mô hình
            self.model = self.build_model()
            
            # Early stopping để tránh overfitting
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=0
            )
            
            # Huấn luyện
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stop],
                verbose=0
            )
            
            # Đánh giá trên test set
            train_predictions = self.model.predict(X_train, verbose=0)
            test_predictions = self.model.predict(X_test, verbose=0)
            
            # Inverse transform để có giá thật
            train_pred_prices = self.scaler.inverse_transform(train_predictions).flatten()
            test_pred_prices = self.scaler.inverse_transform(test_predictions).flatten()
            y_train_prices = self.scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
            y_test_prices = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            
            # Tính metrics
            train_rmse = np.sqrt(np.mean((y_train_prices - train_pred_prices)**2))
            test_rmse = np.sqrt(np.mean((y_test_prices - test_pred_prices)**2))
            train_mae = np.mean(np.abs(y_train_prices - train_pred_prices))
            test_mae = np.mean(np.abs(y_test_prices - test_pred_prices))
            
            self.is_trained = True
            
            return {
                'success': True,
                'metrics': {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'epochs_trained': len(history.history['loss']),
                    'final_loss': history.history['loss'][-1],
                    'final_val_loss': history.history['val_loss'][-1] if 'val_loss' in history.history else None
                },
                'predictions': {
                    'train': train_pred_prices,
                    'test': test_pred_prices,
                    'train_actual': y_train_prices,
                    'test_actual': y_test_prices
                },
                'dates': prices.index[self.lookback_days:],
                'history': history.history
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def predict_future(self, prices: pd.Series) -> dict:
        """Dự báo giá tương lai"""
        if not self.is_trained or self.model is None:
            return {'success': False, 'error': 'LSTM chưa được huấn luyện'}
        
        try:
            # Lấy dữ liệu gần nhất
            recent_data = prices.iloc[-self.lookback_days:].values.reshape(-1, 1)
            scaled_recent = self.scaler.transform(recent_data)
            
            predictions = []
            current_batch = scaled_recent.copy()
            
            # Dự báo từng ngày một cách tuần tự
            for _ in range(self.forecast_days):
                # Reshape cho model (1, timesteps, features)
                current_batch_reshaped = current_batch.reshape(1, self.lookback_days, 1)
                
                # Dự báo
                next_pred = self.model.predict(current_batch_reshaped, verbose=0)[0, 0]
                predictions.append(next_pred)
                
                # Cập nhật batch cho lần dự báo tiếp theo
                current_batch = np.roll(current_batch, -1, axis=0)
                current_batch[-1] = next_pred
            
            # Chuyển đổi về giá thật
            predictions_array = np.array(predictions).reshape(-1, 1)
            future_prices = self.scaler.inverse_transform(predictions_array).flatten()
            
            # Tạo ngày tương lai - Fixed để xử lý mọi loại index
            try:
                # Lấy ngày cuối cùng và convert thành datetime
                last_date_raw = prices.index[-1]
                
                if isinstance(last_date_raw, (int, np.integer)):
                    # Nếu là số, coi như số ngày từ epoch hoặc index position
                    # Tạo ngày từ today
                    last_date = pd.Timestamp.today()
                elif isinstance(last_date_raw, str):
                    # Nếu là string, parse thành datetime
                    last_date = pd.to_datetime(last_date_raw)
                elif hasattr(last_date_raw, 'date') or isinstance(last_date_raw, pd.Timestamp):
                    # Nếu là datetime object
                    last_date = pd.to_datetime(last_date_raw)
                else:
                    # Fallback: sử dụng today
                    last_date = pd.Timestamp.today()
                
                # Tạo future dates
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=self.forecast_days,
                    freq='D'
                )
                
            except Exception as date_error:
                # Nếu có lỗi gì, fallback đơn giản
                print(f"Warning: Date conversion error: {date_error}")
                future_dates = pd.date_range(
                    start=pd.Timestamp.today() + pd.Timedelta(days=1),
                    periods=self.forecast_days,
                    freq='D'
                )
            
            # Khoảng tin cậy dựa trên độ biến động lịch sử
            recent_volatility = prices.iloc[-30:].pct_change().std() if len(prices) >= 30 else 0.02
            uncertainty_base = recent_volatility * np.sqrt(252)  # Annualized volatility
            
            # Tăng uncertainty theo thời gian
            time_factor = np.sqrt(np.arange(1, self.forecast_days + 1) / 252)
            uncertainty = uncertainty_base * time_factor
            
            lower_bound = future_prices * (1 - uncertainty)
            upper_bound = future_prices * (1 + uncertainty)
            
            return {
                'success': True,
                'predictions': future_prices,
                'dates': future_dates,
                'confidence_interval': {
                    'lower': lower_bound,
                    'upper': upper_bound
                },
                'model_info': {
                    'architecture': 'LSTM',
                    'lookback_days': self.lookback_days,
                    'forecast_days': self.forecast_days
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def plot_forecast(self, prices: pd.Series, forecast_results: dict, stock_symbol: str) -> go.Figure:
        """Vẽ biểu đồ dự báo"""
        fig = go.Figure()
        
        if not forecast_results['success']:
            fig.add_annotation(
                text=f"Lỗi: {forecast_results.get('error', 'Unknown')}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Giá lịch sử (100 ngày gần nhất)
        recent_prices = prices.tail(100)
        fig.add_trace(go.Scatter(
            x=recent_prices.index,
            y=recent_prices.values / 1000,  # Chuyển sang nghìn VND
            name='Lịch sử',
            line=dict(color='blue')
        ))
        
        # Dự báo
        fig.add_trace(go.Scatter(
            x=forecast_results['dates'],
            y=forecast_results['predictions'] / 1000,  # Chuyển sang nghìn VND
            name='Dự báo LSTM',
            line=dict(color='red')
        ))
        
        # Khoảng tin cậy
        ci = forecast_results['confidence_interval']
        fig.add_trace(go.Scatter(
            x=forecast_results['dates'],
            y=ci['upper'] / 1000,  # Chuyển sang nghìn VND
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_results['dates'],
            y=ci['lower'] / 1000,  # Chuyển sang nghìn VND
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            fillcolor='rgba(255,0,0,0.2)',
            showlegend=False
        ))
        
        fig.update_layout(
            title=f'Dự báo LSTM {self.forecast_days} ngày - {stock_symbol}',
            xaxis_title='Ngày',
            yaxis_title='Giá (Ngàn VND)',
            showlegend=False,
            height=500
        )
        
        return fig

# Main functions for integration
def run_lstm_analysis(prices_data: pd.Series, stock_symbol: str, lookback_days: int = 60, forecast_days: int = 30) -> dict:
    """Chạy phân tích LSTM"""
    if not TENSORFLOW_AVAILABLE:
        return {
            'success': False,
            'error': 'TensorFlow không có. Vui lòng cài đặt: pip install tensorflow'
        }
    
    try:
        # Làm sạch dữ liệu
        clean_prices = prices_data.dropna()
        
        if len(clean_prices) < lookback_days + 50:
            return {
                'success': False,
                'error': f'Không đủ dữ liệu. Cần {lookback_days + 50} ngày, có {len(clean_prices)} ngày'
            }
        
        # Kiểm tra dữ liệu hợp lệ
        if clean_prices.min() <= 0:
            return {
                'success': False,
                'error': 'Dữ liệu giá không hợp lệ (có giá <= 0)'
            }
        
        predictor = LSTMPredictor(lookback_days, forecast_days)
        
        # Training
        training_results = predictor.train(clean_prices, epochs=30, batch_size=16)
        if not training_results['success']:
            return {
                'success': False,
                'error': f'Training failed: {training_results["error"]}'
            }
        
        # Forecast
        forecast_results = predictor.predict_future(clean_prices)
        if not forecast_results['success']:
            return {
                'success': False,
                'error': f'Forecast failed: {forecast_results["error"]}'
            }
        
        return {
            'success': True,
            'training': training_results,
            'forecast': forecast_results,
            'predictor': predictor
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'LSTM error: {str(e)}'
        }

def simple_moving_average_forecast(prices: pd.Series, forecast_days: int = 30, window: int = 20) -> dict:
    """Simple Moving Average làm backup"""
    try:
        # Làm sạch dữ liệu
        clean_prices = prices.dropna()
        
        if len(clean_prices) < window:
            if len(clean_prices) < 5:
                return {'success': False, 'error': 'Không đủ dữ liệu cho MA'}
            
            trend = np.mean(np.diff(clean_prices.iloc[-min(5, len(clean_prices)):]))
            last_price = clean_prices.iloc[-1]
        else:
            ma = clean_prices.rolling(window=window).mean()
            trend = ma.diff().mean()
            last_price = clean_prices.iloc[-1]
        
        # Giới hạn trend hợp lý
        if abs(trend) > last_price * 0.1:
            trend = np.sign(trend) * last_price * 0.01
        
        predictions = []
        for i in range(forecast_days):
            pred_price = last_price + (trend * (i + 1))
            pred_price = max(pred_price, last_price * 0.5)
            pred_price = min(pred_price, last_price * 1.5)
            predictions.append(pred_price)
        
        # Tạo ngày tương lai - Fixed để xử lý mọi loại index  
        try:
            last_date_raw = clean_prices.index[-1]
            
            if isinstance(last_date_raw, (int, np.integer)):
                # Nếu là số, sử dụng today
                last_date = pd.Timestamp.today()
            elif isinstance(last_date_raw, str):
                # Nếu là string, parse thành datetime
                last_date = pd.to_datetime(last_date_raw)
            elif hasattr(last_date_raw, 'date') or isinstance(last_date_raw, pd.Timestamp):
                # Nếu là datetime object
                last_date = pd.to_datetime(last_date_raw)
            else:
                # Fallback
                last_date = pd.Timestamp.today()
            
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
        except Exception as date_error:
            print(f"Warning: MA date conversion error: {date_error}")
            future_dates = pd.date_range(
                start=pd.Timestamp.today() + pd.Timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
        
        return {
            'success': True,
            'predictions': np.array(predictions),
            'dates': future_dates,
            'confidence_interval': {
                'lower': np.array(predictions) * 0.95,
                'upper': np.array(predictions) * 1.05
            }
        }
        
    except Exception as e:
        return {'success': False, 'error': f'MA error: {str(e)}'}