#!/usr/bin/env python3
"""
Simple Stock Market Data Analyzer
=================================

A simplified version that works with basic Python libraries to analyze
stock market data and provide basic insights without requiring external packages.
"""

import csv
import math
import statistics
from typing import List, Tuple, Dict, Any

class SimpleStockAnalyzer:
    """Simple stock market data analyzer using only standard Python libraries"""
    
    def __init__(self):
        self.data = []
        self.features = []
        self.targets = []
        
    def load_data(self, filename: str) -> bool:
        """
        Load stock market data from CSV file
        
        Args:
            filename (str): Path to CSV file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filename, 'r') as file:
                reader = csv.DictReader(file)
                self.data = list(reader)
            
            print(f"✅ Loaded {len(self.data)} records from {filename}")
            return True
            
        except FileNotFoundError:
            print(f"❌ File not found: {filename}")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Validate the loaded data and return statistics
        
        Returns:
            dict: Validation results and statistics
        """
        if not self.data:
            return {"error": "No data loaded"}
        
        # Extract numeric columns
        numeric_data = {}
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in self.data[0]:
                try:
                    values = [float(row[col]) for row in self.data if row[col]]
                    numeric_data[col] = values
                except ValueError:
                    print(f"⚠️  Warning: Could not convert {col} to numeric")
        
        # Calculate statistics
        stats = {}
        for col, values in numeric_data.items():
            if values:
                stats[col] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0
                }
        
        # Check for data quality issues
        issues = []
        
        # Check for missing values
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in self.data[0]:
                missing = sum(1 for row in self.data if not row[col] or row[col].strip() == '')
                if missing > 0:
                    issues.append(f"{col}: {missing} missing values")
        
        # Check for logical inconsistencies
        if 'High' in numeric_data and 'Low' in numeric_data:
            invalid_high_low = sum(1 for h, l in zip(numeric_data['High'], numeric_data['Low']) if h < l)
            if invalid_high_low > 0:
                issues.append(f"High < Low: {invalid_high_low} invalid entries")
        
        # Check for zero or negative prices
        for col in ['Open', 'Close']:
            if col in numeric_data:
                invalid_prices = sum(1 for price in numeric_data[col] if price <= 0)
                if invalid_prices > 0:
                    issues.append(f"{col}: {invalid_prices} invalid prices (≤0)")
        
        return {
            'total_records': len(self.data),
            'columns': list(self.data[0].keys()) if self.data else [],
            'statistics': stats,
            'issues': issues,
            'data_quality_score': max(0, 100 - len(issues) * 10)
        }
    
    def calculate_technical_indicators(self) -> Dict[str, List[float]]:
        """
        Calculate basic technical indicators
        
        Returns:
            dict: Technical indicators
        """
        if not self.data:
            return {}
        
        # Extract close prices
        close_prices = []
        for row in self.data:
            try:
                close_prices.append(float(row['Close']))
            except (ValueError, KeyError):
                continue
        
        if len(close_prices) < 50:
            print("⚠️  Warning: Not enough data for reliable technical indicators")
            return {}
        
        # Calculate Simple Moving Averages
        sma_10 = self._calculate_sma(close_prices, 10)
        sma_50 = self._calculate_sma(close_prices, 50)
        
        # Calculate Exponential Moving Averages
        ema_10 = self._calculate_ema(close_prices, 10)
        ema_50 = self._calculate_ema(close_prices, 50)
        
        # Calculate price changes
        price_changes = [close_prices[i] - close_prices[i-1] for i in range(1, len(close_prices))]
        
        # Calculate volatility (standard deviation of price changes)
        volatility = statistics.stdev(price_changes) if len(price_changes) > 1 else 0
        
        return {
            'close_prices': close_prices,
            'sma_10': sma_10,
            'sma_50': sma_50,
            'ema_10': ema_10,
            'ema_50': ema_50,
            'price_changes': price_changes,
            'volatility': volatility
        }
    
    def _calculate_sma(self, prices: List[float], window: int) -> List[float]:
        """Calculate Simple Moving Average"""
        sma = []
        for i in range(len(prices)):
            if i < window - 1:
                sma.append(None)
            else:
                sma.append(statistics.mean(prices[i - window + 1:i + 1]))
        return sma
    
    def _calculate_ema(self, prices: List[float], window: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) < window:
            return [None] * len(prices)
        
        alpha = 2.0 / (window + 1)
        ema = [None] * len(prices)
        
        # Initialize with SMA
        ema[window - 1] = statistics.mean(prices[:window])
        
        # Calculate EMA
        for i in range(window, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        
        return ema
    
    def generate_trading_signals(self, indicators: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Generate basic trading signals based on technical indicators
        
        Args:
            indicators (dict): Technical indicators
            
        Returns:
            dict: Trading signals and analysis
        """
        if not indicators or 'close_prices' not in indicators:
            return {"error": "No indicators available"}
        
        close_prices = indicators['close_prices']
        sma_10 = indicators.get('sma_10', [])
        sma_50 = indicators.get('sma_50', [])
        
        signals = []
        signal_reasons = []
        
        for i in range(len(close_prices)):
            signal = "HOLD"
            reason = "No clear signal"
            
            # Simple moving average crossover strategy
            if i >= 50 and sma_10[i] is not None and sma_50[i] is not None:
                if sma_10[i] > sma_50[i] and (i == 0 or sma_10[i-1] <= sma_50[i-1]):
                    signal = "BUY"
                    reason = "SMA 10 crossed above SMA 50"
                elif sma_10[i] < sma_50[i] and (i == 0 or sma_10[i-1] >= sma_50[i-1]):
                    signal = "SELL"
                    reason = "SMA 10 crossed below SMA 50"
            
            # Price momentum
            if i >= 5:
                recent_change = (close_prices[i] - close_prices[i-5]) / close_prices[i-5]
                if recent_change > 0.05:  # 5% increase
                    signal = "BUY"
                    reason = "Strong upward momentum"
                elif recent_change < -0.05:  # 5% decrease
                    signal = "SELL"
                    reason = "Strong downward momentum"
            
            signals.append(signal)
            signal_reasons.append(reason)
        
        # Count signals
        buy_count = signals.count("BUY")
        sell_count = signals.count("SELL")
        hold_count = signals.count("HOLD")
        
        return {
            'signals': signals,
            'signal_reasons': signal_reasons,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'hold_count': hold_count,
            'total_signals': len(signals),
            'buy_percentage': (buy_count / len(signals)) * 100 if signals else 0,
            'sell_percentage': (sell_count / len(signals)) * 100 if signals else 0,
            'hold_percentage': (hold_count / len(signals)) * 100 if signals else 0
        }
    
    def print_analysis_report(self, validation: Dict[str, Any], indicators: Dict[str, List[float]], signals: Dict[str, Any]):
        """
        Print comprehensive analysis report
        
        Args:
            validation (dict): Data validation results
            indicators (dict): Technical indicators
            signals (dict): Trading signals
        """
        print("\n" + "="*60)
        print("STOCK MARKET DATA ANALYSIS REPORT")
        print("="*60)
        
        # Data validation report
        print(f"\n📊 DATA VALIDATION:")
        print(f"   Total records: {validation.get('total_records', 0)}")
        print(f"   Data quality score: {validation.get('data_quality_score', 0)}/100")
        
        if validation.get('issues'):
            print(f"   Issues found: {len(validation['issues'])}")
            for issue in validation['issues']:
                print(f"     ⚠️  {issue}")
        else:
            print("   ✅ No data quality issues found")
        
        # Price statistics
        stats = validation.get('statistics', {})
        if 'Close' in stats:
            close_stats = stats['Close']
            print(f"\n💰 PRICE STATISTICS (Close):")
            print(f"   Price range: ${close_stats['min']:.2f} - ${close_stats['max']:.2f}")
            print(f"   Average price: ${close_stats['mean']:.2f}")
            print(f"   Median price: ${close_stats['median']:.2f}")
            print(f"   Standard deviation: ${close_stats['std']:.2f}")
        
        # Technical indicators
        if indicators:
            print(f"\n📈 TECHNICAL INDICATORS:")
            if 'volatility' in indicators:
                print(f"   Price volatility: {indicators['volatility']:.4f}")
            
            # Show recent SMA values
            if 'sma_10' in indicators and 'sma_50' in indicators:
                sma_10 = indicators['sma_10']
                sma_50 = indicators['sma_50']
                recent_10 = next((x for x in reversed(sma_10) if x is not None), None)
                recent_50 = next((x for x in reversed(sma_50) if x is not None), None)
                
                if recent_10 and recent_50:
                    print(f"   Recent SMA 10: ${recent_10:.2f}")
                    print(f"   Recent SMA 50: ${recent_50:.2f}")
                    
                    if recent_10 > recent_50:
                        print("   📈 SMA 10 > SMA 50 (Bullish trend)")
                    else:
                        print("   📉 SMA 10 < SMA 50 (Bearish trend)")
        
        # Trading signals
        if signals and 'error' not in signals:
            print(f"\n🎯 TRADING SIGNALS:")
            print(f"   Buy signals: {signals['buy_count']} ({signals['buy_percentage']:.1f}%)")
            print(f"   Sell signals: {signals['sell_count']} ({signals['sell_percentage']:.1f}%)")
            print(f"   Hold signals: {signals['hold_count']} ({signals['hold_percentage']:.1f}%)")
            
            # Show recent signals
            recent_signals = signals['signals'][-10:] if len(signals['signals']) >= 10 else signals['signals']
            recent_reasons = signals['signal_reasons'][-10:] if len(signals['signal_reasons']) >= 10 else signals['signal_reasons']
            
            print(f"\n   Recent signals:")
            for i, (signal, reason) in enumerate(zip(recent_signals, recent_reasons)):
                if signal != "HOLD":
                    emoji = "🟢" if signal == "BUY" else "🔴"
                    print(f"     {emoji} {signal}: {reason}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if validation.get('data_quality_score', 0) < 80:
            print("   ⚠️  Data quality issues detected - clean data before trading")
        
        if indicators and 'volatility' in indicators:
            volatility = indicators['volatility']
            if volatility > statistics.mean(indicators['close_prices']) * 0.1:
                print("   ⚠️  High volatility detected - use appropriate risk management")
            else:
                print("   ✅ Volatility levels are reasonable")
        
        if signals and 'error' not in signals:
            signal_frequency = (signals['buy_count'] + signals['sell_count']) / signals['total_signals']
            if signal_frequency > 0.3:
                print("   ⚠️  High signal frequency - consider filtering signals")
            else:
                print("   ✅ Signal frequency is reasonable")
        
        print("\n" + "="*60)


def main():
    """Main function to run the stock analysis"""
    print("🚀 SIMPLE STOCK MARKET ANALYZER")
    print("="*50)
    
    analyzer = SimpleStockAnalyzer()
    
    # Analyze training data
    print("\n📁 Analyzing training data...")
    if analyzer.load_data('stock_market_train.csv'):
        validation = analyzer.validate_data()
        indicators = analyzer.calculate_technical_indicators()
        signals = analyzer.generate_trading_signals(indicators)
        analyzer.print_analysis_report(validation, indicators, signals)
    
    # Analyze test data
    print("\n📁 Analyzing test data...")
    if analyzer.load_data('stock_market_test_final.csv'):
        validation = analyzer.validate_data()
        indicators = analyzer.calculate_technical_indicators()
        signals = analyzer.generate_trading_signals(indicators)
        analyzer.print_analysis_report(validation, indicators, signals)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()