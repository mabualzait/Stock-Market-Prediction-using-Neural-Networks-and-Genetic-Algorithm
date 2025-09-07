function [predictions, signals, performance] = stock_predictor_modern(train_file, test_file, hidden_neurons)
%% Modern Stock Market Prediction System
% This function implements a neural network-based stock market prediction
% system with trading signal generation
%
% Inputs:
%   train_file - Path to training data CSV file
%   test_file - Path to test data CSV file  
%   hidden_neurons - Number of hidden neurons (optional, default: 10)
%
% Outputs:
%   predictions - Predicted closing prices
%   signals - Trading signals (1=buy, 0=hold, -1=sell)
%   performance - Performance metrics (MSE, MAE, Directional Accuracy)

    % Set default parameters
    if nargin < 3
        hidden_neurons = 10;
    end
    
    % Add error handling
    if ~exist(train_file, 'file')
        error('Training file not found: %s', train_file);
    end
    if ~exist(test_file, 'file')
        error('Test file not found: %s', test_file);
    end
    
    fprintf('Loading training data from: %s\n', train_file);
    [train_data, train_targets] = load_stock_data(train_file);
    
    fprintf('Loading test data from: %s\n', test_file);
    [test_data, test_targets] = load_stock_data(test_file);
    
    fprintf('Training neural network with %d hidden neurons...\n', hidden_neurons);
    
    % Create and train neural network
    net = create_neural_network(size(train_data, 1), hidden_neurons);
    net = train(net, train_data, train_targets);
    
    % Make predictions
    fprintf('Making predictions...\n');
    predictions = net(test_data);
    
    % Generate trading signals
    fprintf('Generating trading signals...\n');
    signals = generate_trading_signals(predictions, test_targets);
    
    % Calculate performance metrics
    performance = calculate_performance(predictions, test_targets);
    
    % Display results
    display_results(predictions, test_targets, signals, performance);
    
    % Plot results
    plot_results(test_targets, predictions, signals);
    
end

function [data, targets] = load_stock_data(filename)
%% Load and preprocess stock data
    try
        % Read CSV file
        T = readtable(filename);
        
        % Extract OHLC data
        if ismember('Open', T.Properties.VariableNames)
            open_prices = T.Open;
        else
            open_prices = T{:, 2}; % Assume second column is Open
        end
        
        if ismember('High', T.Properties.VariableNames)
            high_prices = T.High;
        else
            high_prices = T{:, 3}; % Assume third column is High
        end
        
        if ismember('Low', T.Properties.VariableNames)
            low_prices = T.Low;
        else
            low_prices = T{:, 4}; % Assume fourth column is Low
        end
        
        if ismember('Close', T.Properties.VariableNames)
            close_prices = T.Close;
        else
            close_prices = T{:, 5}; % Assume fifth column is Close
        end
        
        % Convert to column vectors
        open_prices = open_prices(:);
        high_prices = high_prices(:);
        low_prices = low_prices(:);
        close_prices = close_prices(:);
        
        % Calculate technical indicators
        sma_10 = movmean(open_prices, 10);
        sma_50 = movmean(open_prices, 50);
        ema_10 = movmean(open_prices, 10, 'Endpoints', 'discard');
        ema_50 = movmean(open_prices, 50, 'Endpoints', 'discard');
        
        % Handle NaN values
        sma_10(isnan(sma_10)) = open_prices(isnan(sma_10));
        sma_50(isnan(sma_50)) = open_prices(isnan(sma_50));
        ema_10(isnan(ema_10)) = open_prices(isnan(ema_10));
        ema_50(isnan(ema_50)) = open_prices(isnan(ema_50));
        
        % Create input matrix (7 features)
        data = [open_prices'; high_prices'; low_prices'; sma_10'; ema_10'; sma_50'; ema_50'];
        targets = close_prices';
        
        % Remove any remaining NaN values
        valid_idx = ~any(isnan(data), 1) & ~isnan(targets);
        data = data(:, valid_idx);
        targets = targets(valid_idx);
        
        fprintf('Loaded %d data points with %d features\n', size(data, 2), size(data, 1));
        
    catch ME
        error('Error loading data from %s: %s', filename, ME.message);
    end
end

function net = create_neural_network(input_size, hidden_neurons)
%% Create a feedforward neural network
    % Create network with modern syntax
    net = feedforwardnet(hidden_neurons);
    
    % Configure training parameters
    net.trainParam.epochs = 1000;
    net.trainParam.goal = 1e-6;
    net.trainParam.lr = 0.01;
    net.trainParam.show = 25;
    net.trainParam.showWindow = false;
    
    % Use all data for training
    net.divideFcn = 'dividetrain';
    
    % Set activation functions
    net.layers{1}.transferFcn = 'tansig';  % Hidden layer
    net.layers{2}.transferFcn = 'purelin'; % Output layer
    
    % Initialize weights
    net = init(net);
end

function signals = generate_trading_signals(predictions, actual_prices)
%% Generate trading signals based on predictions
    signals = zeros(size(predictions));
    
    % Simple signal generation strategy:
    % Buy if prediction > actual * 1.02 (2% higher)
    % Sell if prediction < actual * 0.98 (2% lower)
    % Hold otherwise
    
    for i = 1:length(predictions)
        if predictions(i) > actual_prices(i) * 1.02
            signals(i) = 1;  % Buy signal
        elseif predictions(i) < actual_prices(i) * 0.98
            signals(i) = -1; % Sell signal
        else
            signals(i) = 0;  % Hold signal
        end
    end
end

function performance = calculate_performance(predictions, actual)
%% Calculate performance metrics
    % Mean Squared Error
    mse = mean((predictions - actual).^2);
    
    % Mean Absolute Error
    mae = mean(abs(predictions - actual));
    
    % Root Mean Squared Error
    rmse = sqrt(mse);
    
    % Mean Absolute Percentage Error
    mape = mean(abs((actual - predictions) ./ actual)) * 100;
    
    % Directional Accuracy (percentage of correct direction predictions)
    actual_direction = sign(diff(actual));
    pred_direction = sign(diff(predictions));
    directional_accuracy = sum(actual_direction == pred_direction) / length(actual_direction) * 100;
    
    performance = struct('MSE', mse, 'MAE', mae, 'RMSE', rmse, ...
                        'MAPE', mape, 'DirectionalAccuracy', directional_accuracy);
end

function display_results(predictions, actual, signals, performance)
%% Display prediction results and performance metrics
    fprintf('\n=== STOCK MARKET PREDICTION RESULTS ===\n');
    fprintf('Performance Metrics:\n');
    fprintf('  Mean Squared Error (MSE): %.4f\n', performance.MSE);
    fprintf('  Mean Absolute Error (MAE): %.4f\n', performance.MAE);
    fprintf('  Root Mean Squared Error (RMSE): %.4f\n', performance.RMSE);
    fprintf('  Mean Absolute Percentage Error (MAPE): %.2f%%\n', performance.MAPE);
    fprintf('  Directional Accuracy: %.2f%%\n', performance.DirectionalAccuracy);
    
    % Signal statistics
    buy_signals = sum(signals == 1);
    sell_signals = sum(signals == -1);
    hold_signals = sum(signals == 0);
    
    fprintf('\nTrading Signals Generated:\n');
    fprintf('  Buy signals: %d (%.1f%%)\n', buy_signals, buy_signals/length(signals)*100);
    fprintf('  Sell signals: %d (%.1f%%)\n', sell_signals, sell_signals/length(signals)*100);
    fprintf('  Hold signals: %d (%.1f%%)\n', hold_signals, hold_signals/length(signals)*100);
    
    fprintf('\nPrediction Summary:\n');
    fprintf('  Actual price range: %.2f - %.2f\n', min(actual), max(actual));
    fprintf('  Predicted price range: %.2f - %.2f\n', min(predictions), max(predictions));
    fprintf('  Average actual price: %.2f\n', mean(actual));
    fprintf('  Average predicted price: %.2f\n', mean(predictions));
end

function plot_results(actual, predictions, signals)
%% Plot prediction results and trading signals
    figure('Position', [100, 100, 1200, 800]);
    
    % Plot 1: Actual vs Predicted prices
    subplot(2, 2, 1);
    plot(actual, 'b-', 'LineWidth', 2, 'DisplayName', 'Actual');
    hold on;
    plot(predictions, 'r--', 'LineWidth', 2, 'DisplayName', 'Predicted');
    xlabel('Time Period');
    ylabel('Stock Price');
    title('Actual vs Predicted Stock Prices');
    legend('Location', 'best');
    grid on;
    
    % Plot 2: Prediction error
    subplot(2, 2, 2);
    error = predictions - actual;
    plot(error, 'g-', 'LineWidth', 1.5);
    xlabel('Time Period');
    ylabel('Prediction Error');
    title('Prediction Error Over Time');
    grid on;
    
    % Plot 3: Trading signals
    subplot(2, 2, 3);
    plot(actual, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Price');
    hold on;
    
    % Plot buy signals
    buy_idx = find(signals == 1);
    if ~isempty(buy_idx)
        plot(buy_idx, actual(buy_idx), 'g^', 'MarkerSize', 8, 'MarkerFaceColor', 'g', 'DisplayName', 'Buy');
    end
    
    % Plot sell signals
    sell_idx = find(signals == -1);
    if ~isempty(sell_idx)
        plot(sell_idx, actual(sell_idx), 'rv', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'Sell');
    end
    
    xlabel('Time Period');
    ylabel('Stock Price');
    title('Trading Signals');
    legend('Location', 'best');
    grid on;
    
    % Plot 4: Scatter plot of actual vs predicted
    subplot(2, 2, 4);
    scatter(actual, predictions, 50, 'filled', 'Alpha', 0.6);
    hold on;
    plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--', 'LineWidth', 2);
    xlabel('Actual Price');
    ylabel('Predicted Price');
    title('Actual vs Predicted Scatter Plot');
    grid on;
    
    % Calculate and display R-squared
    r_squared = 1 - sum((actual - predictions).^2) / sum((actual - mean(actual)).^2);
    text(0.05, 0.95, sprintf('R² = %.4f', r_squared), 'Units', 'normalized', ...
         'FontSize', 12, 'BackgroundColor', 'white');
end