function validate_data(filename)
%% Data Validation Script for Stock Market Data
% This function validates the quality and format of stock market data files

    if nargin < 1
        filename = 'stock_market_train.csv';
    end
    
    fprintf('=== DATA VALIDATION REPORT ===\n');
    fprintf('File: %s\n\n', filename);
    
    if ~exist(filename, 'file')
        fprintf('❌ ERROR: File not found\n');
        return;
    end
    
    try
        % Read the data
        T = readtable(filename);
        
        fprintf('✅ File successfully loaded\n');
        fprintf('Data dimensions: %d rows × %d columns\n', height(T), width(T));
        
        % Check column names
        fprintf('\nColumn names:\n');
        for i = 1:width(T)
            fprintf('  %d. %s\n', i, T.Properties.VariableNames{i});
        end
        
        % Check for required columns
        required_cols = {'Date', 'Open', 'High', 'Low', 'Close'};
        missing_cols = {};
        
        for col = required_cols
            if ~ismember(col{1}, T.Properties.VariableNames)
                missing_cols{end+1} = col{1};
            end
        end
        
        if isempty(missing_cols)
            fprintf('\n✅ All required columns present\n');
        else
            fprintf('\n❌ Missing required columns: %s\n', strjoin(missing_cols, ', '));
        end
        
        % Data quality checks
        fprintf('\n=== DATA QUALITY CHECKS ===\n');
        
        % Check for missing values
        if ismember('Open', T.Properties.VariableNames)
            open_missing = sum(isnan(T.Open));
            fprintf('Missing Open values: %d\n', open_missing);
        end
        
        if ismember('High', T.Properties.VariableNames)
            high_missing = sum(isnan(T.High));
            fprintf('Missing High values: %d\n', high_missing);
        end
        
        if ismember('Low', T.Properties.VariableNames)
            low_missing = sum(isnan(T.Low));
            fprintf('Missing Low values: %d\n', low_missing);
        end
        
        if ismember('Close', T.Properties.VariableNames)
            close_missing = sum(isnan(T.Close));
            fprintf('Missing Close values: %d\n', close_missing);
        end
        
        % Check for logical consistency (High >= Low, etc.)
        if ismember('High', T.Properties.VariableNames) && ismember('Low', T.Properties.VariableNames)
            invalid_high_low = sum(T.High < T.Low);
            fprintf('Invalid High < Low entries: %d\n', invalid_high_low);
        end
        
        % Check for zero or negative prices
        if ismember('Open', T.Properties.VariableNames)
            invalid_open = sum(T.Open <= 0);
            fprintf('Invalid Open prices (≤0): %d\n', invalid_open);
        end
        
        if ismember('Close', T.Properties.VariableNames)
            invalid_close = sum(T.Close <= 0);
            fprintf('Invalid Close prices (≤0): %d\n', invalid_close);
        end
        
        % Price statistics
        if ismember('Close', T.Properties.VariableNames)
            fprintf('\n=== PRICE STATISTICS ===\n');
            fprintf('Close price range: %.2f - %.2f\n', min(T.Close), max(T.Close));
            fprintf('Mean close price: %.2f\n', mean(T.Close));
            fprintf('Close price volatility (std): %.2f\n', std(T.Close));
        end
        
        % Date range
        if ismember('Date', T.Properties.VariableNames)
            fprintf('\n=== DATE RANGE ===\n');
            if isdatetime(T.Date)
                fprintf('Date range: %s to %s\n', datestr(min(T.Date)), datestr(max(T.Date)));
            else
                fprintf('Date column found but not in datetime format\n');
            end
        end
        
        fprintf('\n✅ Data validation complete\n');
        
    catch ME
        fprintf('❌ ERROR reading file: %s\n', ME.message);
    end
end