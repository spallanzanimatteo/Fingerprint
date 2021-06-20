function [results] = run_svm_experiment(ca_train, ca_test)

    [x_train, y_train, x_test, ky_test] = prepare_svm_data(ca_train, ca_test);

    pr_y = svm_train_test(x_train, y_train, x_test, cell2mat(ky_test(2:end, 2)));

    results = summarise_experiment_results(ky_test, pr_y);
    inspect_experiment_results(results);

end


function [results] = summarise_experiment_results(ky_test, pr_y)

    results = containers.Map();

    % vector-level results
    raw_vec = horzcat(ky_test, vertcat({'pr_y'}, num2cell(pr_y)));
    raw_vec(1, :) = {'k', 'gt_y', 'pr_y'};
    gt_y = cell2mat(ky_test(2:end, 2));
    cm_vec = confusionmat(gt_y, pr_y);
    results('raw_vec') = raw_vec;
    results('cm_vec') = cm_vec;

    % bag-level results
    k = raw_vec(2:end, 1);
    k_names = unique(k);
    get_k_idx = @(k) (strcmp(raw_vec(2:end, 1), k{1}));
    get_gtpry = @(idx) ([mode(gt_y(idx, :)), mode(pr_y(idx, :))]);

    raw = zeros(length(k_names), 2);
    for ik = 1:length(k_names)
        raw(ik, :) = get_gtpry(get_k_idx(k_names(ik)));
    end
    cm = confusionmat(raw(:, 1), raw(:, 2));
    raw = vertcat(raw_vec(1, :), horzcat(k_names, num2cell(raw)));

    results('raw') = raw;
    results('cm') = cm;

end


function inspect_experiment_results(results)

    fprintf("\nSUPPORT VECTOR MACHINE\n\n");

    fprintf("Vector-level performance\n");
    show_accuracy(results('cm_vec'));

    fprintf("Bag-level performance\n");
    show_accuracy(results('cm'));

end
