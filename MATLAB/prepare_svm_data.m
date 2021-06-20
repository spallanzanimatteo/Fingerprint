function [x_train, y_train, x_test, ky_test] = prepare_svm_data(ca_train, ca_test)

    header = ca_train(1, :);
    k_locs = arrayfun(@(n) startsWith(n, 'k'), header);
    y_locs = arrayfun(@(n) startsWith(n, 'y'), header);

    x_train = ca_train(2:end, ~(k_locs | y_locs));
    x_train = cell2mat(x_train);
    % column scaling in [-1,1]
    max_val = max(abs(x_train), [], 1);
    scaling = ones(1, size(x_train, 2));
    scaling(max_val > 0) = 1 ./ max_val(max_val > 0);
    x_train = x_train * diag(scaling);

    y_train = cell2mat(ca_train(2:end, y_locs));

    x_test = ca_test(2:end, ~(k_locs | y_locs));
    x_test = cell2mat(x_test);
    x_test = x_test * diag(scaling);    

    ky_test = ca_test(:, (k_locs | y_locs));

end
