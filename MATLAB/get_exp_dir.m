function [exp_dir] = get_exp_dir(data_set, data_type)
    exp_dir = fullfile('..', 'Data', data_set, 'Experiments', [upper(data_type(1)), lower(data_type(2:end))]);
end
