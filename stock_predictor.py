#!/usr/bin/env python3
"""
Modern Stock Market Prediction System
====================================

This module implements a neural network-based stock market prediction system
with trading signal generation using modern Python libraries.

Features:
- Neural network-based price prediction
- Trading signal generation (Buy/Sell/Hold)
- Performance metrics calculation
- Data validation and preprocessing
- Interactive visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    """Stock Market Prediction System with Trading Signals"""
    
    def __init__(self, hidden_neurons=10, random_state=42):
        """
        Initialize the stock predictor
        
        Args:
            hidden_neurons (int): Number of hidden neurons in the neural network
            random_state (int): Random state for reproducibility
        """
        self.hidden_neurons = hidden_neurons
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = ['Open', 'High', 'Low', 'SMA_10', 'EMA_10', 'SMA_50', 'EMA_50']
        
    def load_data(self, filename):
        """
        Load and preprocess stock market data
        
        Args:
            filename (str): Path to CSV file
            
        Returns:
            tuple: (features, targets) arrays
        """
        try:
            # Read CSV file
            df = pd.read_csv(filename)
            print(f"✅ Loaded data from {filename}")
            print(f"   Shape: {df.shape}")
            
            # Validate required columns
            required_cols = ['Open', 'High', 'Low', 'Close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Extract OHLC data
            open_prices = df['Open'].values
            high_prices = df['High'].values
            low_prices = df['Low'].values
            close_prices = df['Close'].values
            
            # Calculate technical indicators
            sma_10 = self._calculate_sma(open_prices, 10)
            sma_50 = self._calculate_sma(open_prices, 50)
            ema_10 = self._calculate_ema(open_prices, 10)
            ema_50 = self._calculate_ema(open_prices, 50)
            
            # Create feature matrix
            features = np.column_stack([
                open_prices, high_prices, low_prices,
                sma_10, ema_10, sma_50, ema_50
            ])
            
            # Remove rows with NaN values
            valid_mask = ~np.any(np.isnan(features), axis=1)
            features = features[valid_mask]
            targets = close_prices[valid_mask]
            
            print(f"   Valid data points: {len(targets)}")
            print(f"   Features: {features.shape[1]}")
            
            return features, targets
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def _calculate_sma(self, prices, window):
        """Calculate Simple Moving Average"""
        sma = np.full_like(prices, np.nan)
        for i in range(window - 1, len(prices)):
            sma[i] = np.mean(prices[i - window + 1:i + 1])
        return sma
    
    def _calculate_ema(self, prices, window):
        """Calculate Exponential Moving Average"""
        ema = np.full_like(prices, np.nan)
        alpha = 2.0 / (window + 1)
        
        # Initialize with SMA
        sma = self._calculate_sma(prices, window)
        ema[window - 1] = sma[window - 1]
        
        # Calculate EMA
        for i in range(window, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        
        return ema
    
    def train(self, train_features, train_targets):
        """
        Train the neural network model
        
        Args:
            train_features (np.array): Training features
            train_targets (np.array): Training targets
        """
        print(f"🧠 Training neural network with {self.hidden_neurons} hidden neurons...")
        
        # Scale features
        train_features_scaled = self.scaler.fit_transform(train_features)
        
        # Create and train model
        self.model = MLPRegressor(
            hidden_layer_sizes=(self.hidden_neurons,),
            activation='tanh',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        self.model.fit(train_features_scaled, train_targets)
        print("✅ Training completed")
    
    def predict(self, test_features):
        """
        Make predictions on test data
        
        Args:
            test_features (np.array): Test features
            
        Returns:
            np.array: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        test_features_scaled = self.scaler.transform(test_features)
        predictions = self.model.predict(test_features_scaled)
        return predictions
    
    def generate_signals(self, predictions, actual_prices, threshold=0.02):
        """
        Generate trading signals based on predictions
        
        Args:
            predictions (np.array): Predicted prices
            actual_prices (np.array): Actual prices
            threshold (float): Threshold for signal generation (default: 2%)
            
        Returns:
            np.array: Trading signals (1=Buy, 0=Hold, -1=Sell)
        """
        signals = np.zeros_like(predictions)
        
        for i in range(len(predictions)):
            if predictions[i] > actual_prices[i] * (1 + threshold):
                signals[i] = 1  # Buy signal
            elif predictions[i] < actual_prices[i] * (1 - threshold):
                signals[i] = -1  # Sell signal
            else:
                signals[i] = 0  # Hold signal
        
        return signals
    
    def calculate_performance(self, predictions, actual):
        """
        Calculate performance metrics
        
        Args:
            predictions (np.array): Predicted values
            actual (np.array): Actual values
            
        Returns:
            dict: Performance metrics
        """
        mse = mean_squared_error(actual, predictions)
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        r2 = r2_score(actual, predictions)
        
        # Directional accuracy
        actual_direction = np.sign(np.diff(actual))
        pred_direction = np.sign(np.diff(predictions))
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'Directional_Accuracy': directional_accuracy
        }
    
    def plot_results(self, actual, predictions, signals=None, title="Stock Market Prediction"):
        """
        Plot prediction results and trading signals
        
        Args:
            actual (np.array): Actual prices
            predictions (np.array): Predicted prices
            signals (np.array, optional): Trading signals
            title (str): Plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Plot 1: Actual vs Predicted
        axes[0, 0].plot(actual, 'b-', label='Actual', linewidth=2)
        axes[0, 0].plot(predictions, 'r--', label='Predicted', linewidth=2)
        axes[0, 0].set_xlabel('Time Period')
        axes[0, 0].set_ylabel('Stock Price')
        axes[0, 0].set_title('Actual vs Predicted Prices')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Prediction Error
        error = predictions - actual
        axes[0, 1].plot(error, 'g-', linewidth=1.5)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Time Period')
        axes[0, 1].set_ylabel('Prediction Error')
        axes[0, 1].set_title('Prediction Error Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Trading Signals (if provided)
        if signals is not None:
            axes[1, 0].plot(actual, 'b-', label='Price', linewidth=1.5)
            
            # Plot buy signals
            buy_idx = np.where(signals == 1)[0]
            if len(buy_idx) > 0:
                axes[1, 0].scatter(buy_idx, actual[buy_idx], color='green', 
                                 marker='^', s=100, label='Buy', zorder=5)
            
            # Plot sell signals
            sell_idx = np.where(signals == -1)[0]
            if len(sell_idx) > 0:
                axes[1, 0].scatter(sell_idx, actual[sell_idx], color='red', 
                                 marker='v', s=100, label='Sell', zorder=5)
            
            axes[1, 0].set_xlabel('Time Period')
            axes[1, 0].set_ylabel('Stock Price')
            axes[1, 0].set_title('Trading Signals')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No signals provided', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Trading Signals')
        
        # Plot 4: Scatter plot
        axes[1, 1].scatter(actual, predictions, alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val = min(min(actual), min(predictions))
        max_val = max(max(actual), max(predictions))
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        axes[1, 1].set_xlabel('Actual Price')
        axes[1, 1].set_ylabel('Predicted Price')
        axes[1, 1].set_title('Actual vs Predicted Scatter')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_performance_report(self, performance, signals=None):
        """
        Print detailed performance report
        
        Args:
            performance (dict): Performance metrics
            signals (np.array, optional): Trading signals
        """
        print("\n" + "="*50)
        print("STOCK MARKET PREDICTION PERFORMANCE REPORT")
        print("="*50)
        
        print(f"\n📊 PREDICTION ACCURACY:")
        print(f"   Mean Squared Error (MSE):     {performance['MSE']:.4f}")
        print(f"   Mean Absolute Error (MAE):    {performance['MAE']:.4f}")
        print(f"   Root Mean Squared Error:      {performance['RMSE']:.4f}")
        print(f"   Mean Absolute % Error (MAPE): {performance['MAPE']:.2f}%")
        print(f"   R-squared (R²):               {performance['R2']:.4f}")
        print(f"   Directional Accuracy:         {performance['Directional_Accuracy']:.2f}%")
        
        if signals is not None:
            buy_signals = np.sum(signals == 1)
            sell_signals = np.sum(signals == -1)
            hold_signals = np.sum(signals == 0)
            total_signals = len(signals)
            
            print(f"\n📈 TRADING SIGNALS:")
            print(f"   Buy signals:  {buy_signals:3d} ({buy_signals/total_signals*100:.1f}%)")
            print(f"   Sell signals: {sell_signals:3d} ({sell_signals/total_signals*100:.1f}%)")
            print(f"   Hold signals: {hold_signals:3d} ({hold_signals/total_signals*100:.1f}%)")
            
            print(f"\n🎯 TRADING RECOMMENDATIONS:")
            if performance['Directional_Accuracy'] > 60:
                print("   ✅ Good directional accuracy - suitable for trading")
            elif performance['Directional_Accuracy'] > 50:
                print("   ⚠️  Moderate directional accuracy - use with caution")
            else:
                print("   ❌ Poor directional accuracy - not recommended for trading")
            
            if performance['MAPE'] < 5:
                print("   ✅ Good prediction accuracy (MAPE < 5%)")
            elif performance['MAPE'] < 10:
                print("   ⚠️  Moderate prediction accuracy (MAPE < 10%)")
            else:
                print("   ❌ Poor prediction accuracy (MAPE > 10%)")
            
            signal_frequency = (buy_signals + sell_signals) / total_signals
            if signal_frequency > 0.3:
                print("   ⚠️  High signal frequency - consider filtering")
            else:
                print("   ✅ Reasonable signal frequency")


def main():
    """Main function to run the stock prediction system"""
    print("🚀 STOCK MARKET PREDICTION SYSTEM")
    print("="*50)
    
    # Initialize predictor
    predictor = StockPredictor(hidden_neurons=10)
    
    try:
        # Load training data
        print("\n📁 Loading training data...")
        train_features, train_targets = predictor.load_data('stock_market_train.csv')
        
        # Load test data
        print("\n📁 Loading test data...")
        test_features, test_targets = predictor.load_data('stock_market_test_final.csv')
        
        # Train model
        print("\n🧠 Training model...")
        predictor.train(train_features, train_targets)
        
        # Make predictions
        print("\n🔮 Making predictions...")
        predictions = predictor.predict(test_features)
        
        # Generate trading signals
        print("\n📊 Generating trading signals...")
        signals = predictor.generate_signals(predictions, test_targets)
        
        # Calculate performance
        performance = predictor.calculate_performance(predictions, test_targets)
        
        # Print report
        predictor.print_performance_report(performance, signals)
        
        # Plot results
        print("\n📈 Generating plots...")
        predictor.plot_results(test_targets, predictions, signals)
        
        print("\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main()