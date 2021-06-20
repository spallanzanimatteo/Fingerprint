function svm_main(data_set, data_type, save)

    [ca_train, ca_test] = load_data(data_set, data_type);

    buffer = fullfile('.', 'SVM');  % temporary files produced by GPDT will be stored here
    if ~exist(buffer, 'dir')
        mkdir(buffer);
    end
    try
        results = run_svm_experiment(ca_train, ca_test);
        success = true;
    catch
        fprintf("Error in function `run_svm_experiment`.");
        success = false;
    end
    if (save && success)
        save_svm_results(buffer, results);
        save_dir = fullfile(get_exp_dir(data_set, data_type), 'SVM');
        [~, ~, ~] = copyfile(buffer, save_dir);
    end
    rmdir(buffer, 's');
    
end
