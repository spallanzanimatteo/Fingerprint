function [pr_y] = svm_train_test(x_train, y_train, x_test, y_test)

    buffer = fullfile('.', 'SVM');  % temporary files produced by GPDT will be stored here

    %% Data set-up

    % parameters of the classification problem
    classes = sort(unique(y_train));
    nY = length(classes);
    for iy = 1:nY  % prepare info to build one-vs-one training sets
        ind{iy, 1} = (y_train == classes(iy));
        ind{iy, 2} = sum(ind{iy, 1});
    end
    classifiers = nchoosek(classes, 2);
    n_models = size(classifiers, 1);

    %% Training

    % write data in SVMlight format
    temp_train_file = fullfile(buffer, 'temp_train');
    create_SVMlight_data(temp_train_file, x_train, y_train);
    x_train = dataread('file', temp_train_file, '%s', 'delimiter', '\n');  % training data must be read to prepare fold data sets

    % cross-validation setup (to tune C and gamma hyper-parameters)
    n_folds = 3;
    C_candidates = [1, 10, 100];
    n_C = length(C_candidates);
    g_candidates = [0.05, 0.1, 0.5];
    n_g = length(g_candidates);

    hyperparams_file = fullfile(buffer, 'hyper-parameters.txt');  % log the best hyper-parameters for each model
    fid_hpf = fopen(hyperparams_file, 'w');

    % build one-vs-one models
    for im = 1:n_models

        % identify model
        class1 = classifiers(im, 1);
        class2 = classifiers(im, 2);
        fprintf(fid_hpf, 'Classifier %d vs %d\n', class1, class2);

        % select and prepare model data
        x_temp = x_train(ind{class1, 1} | ind{class2, 1});
        for ix = 1:(ind{class1, 2} + ind{class2, 2})
            if (x_temp{ix}(1) == num2str(class1))
                x_temp{ix} = ['-1' x_temp{ix}(2:end)];
            else
                x_temp{ix} = ['1' x_temp{ix}(2:end)];
            end
        end

        % prepare labels of model data(to estimate/compute model accuracy)
        y_temp = y_train(ind{class1, 1} | ind{class2, 1});
        y_temp(y_temp == class1) = -1;
        y_temp(y_temp == class2) = 1;

        % cross-validation?
        if (n_C > 1) || (n_g > 1)

            fold_train_file = fullfile(buffer, 'fold_train_data');
            fold_test_file  = fullfile(buffer, 'fold_test_data');
            fold_model_file = fullfile(buffer, 'fold_model');
            fold_preds_file = fullfile(buffer, 'fold_preds');

            err_cv = zeros(n_folds, n_C, n_g);  % count the errors of each model

            % split training points in folds (sequentially)
            n_x_temp = length(x_temp);
            fold_sizes = [ones(1, n_folds - 1) * floor(n_x_temp / n_folds), n_x_temp - (n_folds - 1) * floor(n_x_temp / n_folds)];

            for i_fold = 1:n_folds

                fold_test_idx = sum(fold_sizes(1:i_fold-1)) + (1:fold_sizes(i_fold));

                % select test fold
                fid = fopen(fold_test_file, 'w');
                for ix = fold_test_idx
                    fprintf(fid, '%s\n', x_temp{ix, :});
                end
                fclose(fid);

                % merge training folds
                fid = fopen(fold_train_file, 'w');
                for ix = setdiff(1:n_x_temp, fold_test_idx)
                    fprintf(fid, '%s\n', x_temp{ix, :});
                end
                fclose(fid);

                % train SVMs
                for i_C = 1:n_C
                    for i_g = 1:n_g

                        % train
                        eval(['!gpdt -t 2 -g ' num2str(g_candidates(i_g)) ' -c ' num2str(C_candidates(i_C)) ' -m 400 -q 1000 -v 0 ' fold_train_file ' ' fold_model_file]);

                        % evaluate
                        eval(['!classify ' fold_test_file ' ' fold_model_file ' ' fold_preds_file]);
                        fold_preds = load(fold_preds_file);
                        err_cv(i_fold, i_C, i_g) = sum(abs(sign(fold_preds) - y_temp(fold_test_idx))) / 2;

                    end
                end

            end

            % determine optimal parameters
            err_cv = squeeze(mean(err_cv));
            [~, i_opt] = min(err_cv(:));
            [i_C, i_g] = ind2sub(size(err_cv), i_opt(1));
            C_opt = C_candidates(i_C);
            g_opt = g_candidates(i_g);
            fprintf(fid_hpf, 'C_opt = %g, g_opt = %g\n', C_opt, g_opt);

            delete(fold_train_file);
            delete(fold_test_file);
            delete(fold_model_file);
            delete(fold_preds_file);

        else

            C_opt = C_candidates;
            g_opt = g_candidates;

        end

        % retrain SVM using all folds and optimal parameters
        temp_1vs1_file = fullfile(buffer, 'temp_1vs1');  % buffer file for one-vs-one training sets
        fid = fopen(temp_1vs1_file, 'w');
        for ix = 1:n_x_temp
            fprintf(fid, '%s\n', x_temp{ix, :});
        end
        fclose(fid);

        model_file = fullfile(buffer, strcat(['model_', num2str(class1), '_', num2str(class2)]));
        eval(['!gpdt -t 2 -g ' num2str(g_opt) ' -c ' num2str(C_opt) ' -m 400 -q 1000 -v 0 ' temp_1vs1_file ' ' model_file]);
    
        delete(temp_1vs1_file);  % free buffer allocated to one-vs-one training sets

    end

    fclose(fid_hpf);

    delete(temp_train_file);

    %% Test (vector-level classification)

    % write data in SVMlight format
    temp_test_file = fullfile(buffer, 'temp_test');
    create_SVMlight_data(temp_test_file, x_test, y_test);

    preds = zeros(length(x_test), length(classifiers));

    for im = 1:size(classifiers, 1)

        class1 = classifiers(im, 1);
        class2 = classifiers(im, 2);

        temp_preds_file = fullfile(buffer, strcat(['temp_preds_', num2str(class1), '_', num2str(class2)]));

        model_file = fullfile(buffer, strcat(['model_', num2str(class1), '_', num2str(class2)]));
        eval(['!classify ' temp_test_file ' ' model_file ' ' temp_preds_file]);

        preds(:, im) = sign(load(temp_preds_file));
        preds(preds(:, im) == 1, im) = class2;
        preds(preds(:, im) == -1, im) = class1;  % the order of the assignments is CRITICAL: what would happen if the lines were swapped and `class1 == 1`?

        delete(temp_preds_file);

    end

    pr_y = mode(preds, 2);
    
    delete(temp_test_file);

end


function [] = create_SVMlight_data(f_light, x, y)

    fid = fopen(f_light, 'w');
    for i = 1:size(x, 1)

        fprintf(fid, '%g', y(i));
        iidx = find(x(i, :) ~= 0);
        for j = 1:length(iidx)
            fprintf(fid, ' %g%s%g', iidx(j), ':', x(i, iidx(j)));
        end
        fprintf(fid, '\n');

    end
    fclose(fid);

end
