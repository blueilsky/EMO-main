clc; clear; close all;
% datafiles = {'new-thyroid1.csv','new-thyroid2.csv'}; 
dataFolder = 'data';
fileStructs = dir(fullfile(dataFolder, '*.csv'));
datafiles = {fileStructs.name};
classifiers = {'RF','DecisionTree', 'NaiveBayes'};

methods = { 'SMOTE','BorderlineSMOTE1','BorderlineSMOTE2', 'SafeLevelSMOTE','SMOTEWB','MWSMOTE','G_SMOTE','WGAN','CTGAN','MEB_Border'};
K = 5;  

metricNames = {'F1', 'MCC','ACC'};

allResults = struct();
nRuns = 10;
allRuns = cell(1, nRuns); 

for run = 1:nRuns
    for d = 1:length(datafiles)
        fprintf('\n=== Processing dataset: %s ===\n', datafiles{d});
        dsName = matlab.lang.makeValidName(datafiles{d});
        data = readtable(datafiles{d});
        X = data{:,1:end-1};
        y_raw = data{:, end};
        y = convertLabelToBinary(y_raw);
        X = zscore(X);
%         X_min = min(X, [], 1, 'omitnan');
%         X_max = max(X, [], 1, 'omitnan');
%         range = X_max - X_min;
%         range(range == 0) = 1;  
%         X = (X - X_min) ./ range;

        
        cv = cvpartition(y, 'KFold', K);
        
        for c = 1:length(classifiers)
            clfName = classifiers{c};
            
            for methodIdx = 1:length(methods)
                method = methods{methodIdx};
                fprintf('Classifier: %s | Method: %s\n', clfName, method);
                
                metricsMat = zeros(K, length(metricNames));
                
                for fold = 1:K
                    train_idx = training(cv, fold);
                    test_idx = test(cv, fold);
                    X_train = X(train_idx,:);
                    y_train = y(train_idx);
                    X_test = X(test_idx,:);
                    y_test = y(test_idx);
                    
                    switch method
                        case 'SMOTE'
                            [X_train_res, y_train_res] = smote_custom(X_train, y_train);
                        case 'BorderlineSMOTE1'
                            [X_train_res, y_train_res] = python_resample('BorderlineSMOTE1', X_train, y_train);
                        case 'BorderlineSMOTE2'
                            [X_train_res, y_train_res] = python_resample('BorderlineSMOTE2', X_train, y_train);
                        case 'SafeLevelSMOTE'
                            [X_train_res, y_train_res] = python_resample('SafeLevelSMOTE', X_train, y_train);
                        case 'SMOTEWB'
                            [X_train_res, y_train_res] = python_resample('SMOTEWB', X_train, y_train);
                        case 'MWSMOTE'
                            [X_train_res, y_train_res] = python_resample('MWSMOTE', X_train, y_train);

                        case 'G_SMOTE'
                            [X_train_res, y_train_res] = python_resample('G_SMOTE', X_train, y_train);
                            
                        case 'WGAN'
                            [X_train_res, y_train_res] = python_resample_new('WGAN', X_train, y_train);
                       
                        case 'CTGAN'
                            [X_train_res, y_train_res] = python_resample_new('CTGAN', X_train, y_train);
                        case 'MEB_Border'
                            [X_train_res, y_train_res] = meb_plot(X_train, y_train);
    
                    end
                    
                    y_train_res = double(y_train_res);
                    y_test = double(y_test);
                    X_train_res(~isfinite(X_train_res)) = NaN;
                    X_test(~isfinite(X_test)) = NaN;
                    
                    train_mean = mean(X_train_res, 'omitnan');
                    test_mean = mean(X_test, 'omitnan');
                    for i = 1:size(X_train_res, 2)
                        nanIdx = isnan(X_train_res(:, i));
                        X_train_res(nanIdx, i) = train_mean(i);
                        
                        nanIdx_test = isnan(X_test(:, i));
                        X_test(nanIdx_test, i) = test_mean(i);
                    end

                    X_train_res(X_train_res > 1e6) = 1e6;
                    X_train_res(X_train_res < -1e6) = -1e6;
                    X_test(X_test > 1e6) = 1e6;
                    X_test(X_test < -1e6) = -1e6;


                    X_train_res(~isfinite(X_train_res)) = 0;
                    X_train_res(isnan(X_train_res)) = 0;
                    X_test(~isfinite(X_test)) = 0;
                    X_test(isnan(X_test)) = 0;

                    [pred, score] = python_classifier_predict(clfName, X_train_res, y_train_res, X_test);

                    confmat = confusionmat(y_test, pred);
                    TP = confmat(2,2); TN = confmat(1,1);
                    FP = confmat(1,2); FN = confmat(2,1);
                    % Accuracy
                    acc = (TP + TN) / (TP + TN + FP + FN + eps);

                    % F1
                    f1 = 2*(precision*recall)/(precision+recall+eps);

                    % MCC
                    numerator = TP*TN - FP*FN;
                    denominator = sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) + eps;
                    mcc = numerator/denominator;
                    
                    metricsMat(fold,:) = [f1, mcc,acc];
                end
                
                allResults.(dsName).(clfName).(method) = metricsMat;
            end
        end
    end
    allRuns{run} = allResults;  
end

outputFolder = 'result';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

for k = 1:length(metricNames)
    metric = metricNames{k};  
    for c = 1:length(classifiers)
        clfName = classifiers{c}; 

        resultTable = cell(length(datafiles), length(methods)+1);
        resultTable(:,1) = datafiles';  

        for m = 1:length(methods)
            method = methods{m};
            for d = 1:length(datafiles)
                dsName = matlab.lang.makeValidName(datafiles{d});
                scores = zeros(nRuns, 1);

                for run = 1:nRuns
                    try
                        mat = allRuns{run}.(dsName).(clfName).(method);
                        scores(run) = mean(mat(:,k), 'omitnan');
                    catch
                        scores(run) = NaN;
                    end
                end

                resultTable{d, m+1} = mean(scores, 'omitnan'); 
            end
        end
        header = [{'Dataset'}, methods];
        T = cell2table(resultTable, 'VariableNames', header);

        filename = sprintf('%s/%s_%s.csv', outputFolder, clfName, metric);
        writetable(T, filename);
        fprintf('保存：%s\n', filename);
    end
end



