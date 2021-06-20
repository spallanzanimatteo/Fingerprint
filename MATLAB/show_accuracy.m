function show_accuracy(cm)

    fprintf("Confusion matrix:\n");
    disp(cm);

    per_class = diag(cm) ./ sum(cm, 2);
    for iy = 1:size(cm, 1)
        fprintf("Accuracy on class %d: \t%6.2f%% \n", iy, 100.0 * per_class(iy));
    end

    overall = sum(diag(cm)) / sum(cm, 'all');
    fprintf("Overall accuracy: \t\t%6.2f%% \n", 100.0 * overall);

end
