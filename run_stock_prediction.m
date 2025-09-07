%% Stock Market Prediction System - Main Interface
% This script provides an easy-to-use interface for stock market prediction
% and trading signal generation

clear; clc; close all;

fprintf('=== STOCK MARKET PREDICTION SYSTEM ===\n');
fprintf('Loading and analyzing stock market data...\n\n');

% Check if data files exist
if ~exist('stock_market_train.csv', 'file')
    error('Training data file not found. Please ensure stock_market_train.csv is in the current directory.');
end

if ~exist('stock_market_test_final.csv', 'file')
    error('Test data file not found. Please ensure stock_market_test_final.csv is in the current directory.');
end

% Run prediction with different network configurations
hidden_neurons_options = [5, 10, 15, 20];
best_performance = inf;
best_neurons = 0;
best_results = [];

fprintf('Testing different neural network configurations...\n');

for i = 1:length(hidden_neurons_options)
    neurons = hidden_neurons_options(i);
    fprintf('\n--- Testing with %d hidden neurons ---\n', neurons);
    
    try
        [predictions, signals, performance] = stock_predictor_modern(...
            'stock_market_train.csv', 'stock_market_test_final.csv', neurons);
        
        % Store best performing configuration
        if performance.MSE < best_performance
            best_performance = performance.MSE;
            best_neurons = neurons;
            best_results = struct('predictions', predictions, 'signals', signals, 'performance', performance);
        end
        
        fprintf('MSE: %.4f, Directional Accuracy: %.2f%%\n', ...
                performance.MSE, performance.DirectionalAccuracy);
        
    catch ME
        fprintf('Error with %d neurons: %s\n', neurons, ME.message);
    end
end

% Display best results
if ~isempty(best_results)
    fprintf('\n=== BEST CONFIGURATION ===\n');
    fprintf('Optimal hidden neurons: %d\n', best_neurons);
    fprintf('Best MSE: %.4f\n', best_results.performance.MSE);
    fprintf('Best Directional Accuracy: %.2f%%\n', best_results.performance.DirectionalAccuracy);
    
    % Generate final report
    generate_trading_report(best_results.predictions, best_results.signals, best_results.performance);
    
    % Save results
    save('stock_prediction_results.mat', 'best_results', 'best_neurons');
    fprintf('\nResults saved to stock_prediction_results.mat\n');
else
    fprintf('\nNo successful predictions were made. Please check your data files.\n');
end

fprintf('\n=== ANALYSIS COMPLETE ===\n');

%% Trading Report Generation Function
function generate_trading_report(predictions, signals, performance)
    fprintf('\n=== TRADING SIGNAL ANALYSIS ===\n');
    
    % Calculate potential returns
    buy_signals = find(signals == 1);
    sell_signals = find(signals == -1);
    
    if length(buy_signals) > 1
        fprintf('Buy signals detected at positions: %s\n', mat2str(buy_signals(1:min(10, end))));
        if length(buy_signals) > 10
            fprintf('... and %d more buy signals\n', length(buy_signals) - 10);
        end
    end
    
    if length(sell_signals) > 1
        fprintf('Sell signals detected at positions: %s\n', mat2str(sell_signals(1:min(10, end))));
        if length(sell_signals) > 10
            fprintf('... and %d more sell signals\n', length(sell_signals) - 10);
        end
    end
    
    % Risk assessment
    prediction_volatility = std(predictions);
    fprintf('\nRisk Assessment:\n');
    fprintf('  Prediction volatility: %.4f\n', prediction_volatility);
    
    if performance.DirectionalAccuracy > 60
        fprintf('  Risk Level: LOW (Good directional accuracy)\n');
    elseif performance.DirectionalAccuracy > 50
        fprintf('  Risk Level: MEDIUM (Moderate directional accuracy)\n');
    else
        fprintf('  Risk Level: HIGH (Poor directional accuracy)\n');
    end
    
    % Trading recommendations
    fprintf('\nTrading Recommendations:\n');
    if performance.MAPE < 5
        fprintf('  ✓ Model shows good accuracy (MAPE < 5%%)\n');
    else
        fprintf('  ⚠ Model accuracy could be improved (MAPE > 5%%)\n');
    end
    
    if performance.DirectionalAccuracy > 55
        fprintf('  ✓ Directional accuracy is acceptable for trading\n');
    else
        fprintf('  ⚠ Consider improving model before live trading\n');
    end
    
    if length(buy_signals) + length(sell_signals) > length(signals) * 0.3
        fprintf('  ⚠ High signal frequency - consider filtering\n');
    else
        fprintf('  ✓ Signal frequency is reasonable\n');
    end
end