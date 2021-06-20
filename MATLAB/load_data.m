function [ca_train, ca_test] = load_data(data_set, data_type)

    data_file = fullfile('..', 'Data', data_set, 'data_set.xlsx');
    [~, ~, ca_train] = xlsread(data_file, 'training');
    [~, ~, ca_test] = xlsread(data_file, 'test');

    if strcmp(data_type, 'numerical')
        ca_train = ca_train(:, ~startsWith(ca_train(1, :), 'q'));
        ca_test = ca_test(:, ~startsWith(ca_test(1, :), 'q'));
    else if strcmp(data_type, 'categorical')
        ca_train = ca_train(:, ~startsWith(ca_train(1, :), 'x'));
        ca_test = ca_test(:, ~startsWith(ca_test(1, :), 'x'));
    end

end
