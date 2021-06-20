library(MASS)


prepare_lda_data <- function(dt_train, dt_test)
{
  x_train <- dt_train[, !c('k', 'y')]
  y_train <- dt_train$y

  x_test <- dt_test[, !c('k', 'y')]
  ky_test <- dt_test[, c('k', 'y')]

  ready_data <- list('x_train'=x_train, 'y_train'=y_train, 'x_test'=x_test, 'ky_test'=ky_test)
  return(ready_data)
}


summarise_lda_experiment_results <- function(model, ky_test, pr_y)
{
  classes <- as.factor(model$lev)

  raw_vec <- cbind(ky_test, pr_y)
  names(raw_vec) <- c('k', 'gt_y', 'pr_y')
  cm_vec <- table(factor(raw_vec$gt_y, classes), factor(raw_vec$pr_y, classes))
  
  raw <- raw_vec[, .(get_mode(gt_y), get_mode(pr_y)), by='k']
  names(raw) <- names(raw_vec)
  cm <- table(factor(raw$gt_y, classes), factor(raw$pr_y, classes))
  
  results <- list('model'=model, 'raw'=raw, 'cm'=cm, 'raw_vec'=raw_vec, 'cm_vec'=cm_vec)
  return(results)
}


inspect_lda_experiment_results <- function(results)
{
  cat(sprintf("\nLINEAR DISCRIMINANT ANALYSIS\n\n"))
  
  cat(sprintf("Vector-level performance\n"))
  show_accuracy(as.factor(results$model$lev), results$cm_vec)
  
  cat(sprintf("Bag-level performance\n"))
  show_accuracy(as.factor(results$model$lev), results$cm)
}


run_lda_experiment <- function(dt_train, dt_test)
{
  ready_data <- prepare_lda_data(dt_train, dt_test)
  x_train <- ready_data$x_train
  y_train <- ready_data$y_train
  x_test <- ready_data$x_test
  ky_test <- ready_data$ky_test
  
  lda_model <- lda(x_train, y_train)
  pr_y <- predict(lda_model, x_test)$class
  
  results <- summarise_lda_experiment_results(lda_model, ky_test, pr_y)
  inspect_lda_experiment_results(results)
  
  return(results)
}


save_lda_results <- function(save_dir, results)
{
  dir.create(save_dir, recursive=TRUE, showWarnings=FALSE)
  
  # R model
  model <- results$model
  save(model, file=file.path(save_dir, 'model.Rmodel'))

  # bag-level performance
  wb <- createWorkbook()
  addWorksheet(wb, 'raw')
  addWorksheet(wb, 'cm')
  writeData(wb, 'raw', results$raw)
  class(results$cm) <- 'matrix'  # contingency tables are stored with an "index" column in the Excel file, even with `rowNames == FALSE`
  writeData(wb, 'cm', as.data.frame(results$cm), colNames=FALSE, rowNames=FALSE)
  saveWorkbook(wb, file=file.path(save_dir, 'results.xlsx'), overwrite=TRUE)
    
  # vector-level performance
  wb_vec <- createWorkbook()
  addWorksheet(wb_vec, 'raw')
  addWorksheet(wb_vec, 'cm')
  writeData(wb_vec, 'raw', results$raw_vec)
  class(results$cm_vec) <- 'matrix'  # contingency tables are stored with an "index" column in the Excel file, even with `rowNames == FALSE`
  writeData(wb_vec, 'cm', as.data.frame(results$cm_vec), colNames=FALSE, rowNames=FALSE)
  saveWorkbook(wb_vec, file=file.path(save_dir, 'results_vec.xlsx'), overwrite=TRUE)
}
