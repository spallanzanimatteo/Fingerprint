function save_svm_results(save_dir, results)

    xlsxfile = fullfile(save_dir, 'results.xlsx');
    xlswrite(xlsxfile, results('raw'), 'raw');
    xlswrite(xlsxfile, results('cm'), 'cm');
    
    xlsxfile = fullfile(save_dir, 'results_vec.xlsx');
    xlswrite(xlsxfile, results('raw_vec'), 'raw');
    xlswrite(xlsxfile, results('cm_vec'), 'cm');
    
end
