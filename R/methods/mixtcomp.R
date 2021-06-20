library(RMixtComp)


onehot2factor <- function(dt)
{
  q_names <- names(dt)[startsWith(names(dt), 'q')]
  
  if (length(q_names) > 0)
  {
    q_factors <- sort(unique(sapply(strsplit(q_names, '_'), '[', 1)))
    
    for (iq in 1:length(q_factors))
    {
      qi_names <- names(dt)[startsWith(names(dt), q_factors[iq])]
      qi <- apply(dt[, ..qi_names], 1, which.max)  # convert to integer modalities
      dt <- dt[, !..qi_names]  # drop one-hot encodings
      dt <- dt[, (q_factors[iq]):=qi]
    }
    
    x_names <- names(dt)[startsWith(names(dt), 'x')]
    setcolorder(dt, c('k', q_factors, x_names, 'y'))
  }
  
  return(dt)
}


prepare_mc_data <- function(dt_train, dt_test)
{
  dt_train <- onehot2factor(dt_train)
  dt_test <- onehot2factor(dt_test)
  
  dt_train <- dt_train[, !c('k')]
  ky_test <- dt_test[, c('k', 'y')]
  dt_test <- dt_test[, !c('k', 'y')]
  # dt_test[['y']] <- NULL  # don't complete latent variable, otherwise the MixtComp software will use the existing values!

  names(dt_train)[names(dt_train) == 'y'] <- 'z_class'
  names(dt_test)[names(dt_test) == 'y'] <- 'z_class'
  
  ready_data <- list('dt_train'=dt_train, 'dt_test'=dt_test, 'ky_test'=ky_test)
  return(ready_data)
}


get_mixtcomp_model <- function(dt)
{
  attrs <- names(dt)
  
  mcm <- list()
  for (i in 1:length(attrs))
  {
    if (startsWith(attrs[i], 'q'))
    {
      mcm[[attrs[i]]] <- list(type='Multinomial', paramStr='')
    }
    else if (startsWith(attrs[i], 'x'))
    {
      mcm[[attrs[i]]] <- list(type='Gaussian', paramStr='')
    }
    else if (attrs[i] == 'z_class')
    {
      mcm[[attrs[i]]] <- list(type='LatentClass', paramStr='')
    }
  }
  
  return(mcm)
}


summarise_mc_experiment_results <- function(model, ky_test, pr_y)
{
  classes <- as.factor(sort(unique(model$variable$data$z_class$completed)))
  
  raw_vec <- cbind(ky_test, pr_y)
  names(raw_vec) <- c('k', 'gt_y', 'pr_y')
  cm_vec <- table(factor(raw_vec$gt_y, classes), factor(raw_vec$pr_y, classes))
  
  raw <- raw_vec[, .(get_mode(gt_y), get_mode(pr_y)), by='k']
  names(raw) <- names(raw_vec)
  cm <- table(factor(raw$gt_y, classes), factor(raw$pr_y, classes))
  
  results <- list('model'=model, 'raw'=raw, 'cm'=cm, 'raw_vec'=raw_vec, 'cm_vec'=cm_vec)
  return(results)
}


inspect_mc_experiment_results <- function(results)
{
  classes <- as.factor(sort(unique(results$model$variable$data$z_class$completed)))

  cat(sprintf("\nMIXTCOMP\n\n"))
  
  cat(sprintf("Vector-level performance\n"))
  show_accuracy(classes, results$cm_vec)
  
  cat(sprintf("Bag-level performance\n"))
  show_accuracy(classes, results$cm)
}


run_mc_experiment <- function(dt_train, dt_test, verbose=FALSE)
{
  ready_data <- prepare_mc_data(dt_train, dt_test)
  dt_train <- ready_data$dt_train
  dt_test <- ready_data$dt_test
  ky_test <- ready_data$ky_test
  
  Sys.setenv(MC_DETERMINISTIC=42)  # for replicability, since MixtComp uses stochastic sampling under the hood
  mcm <- get_mixtcomp_model(dt_train)
  algo <- createAlgo(nInitPerClass=length(dt_train$z_class))
  mc_model <- mixtCompLearn(dt_train, model=mcm, algo=algo, nClass=length(unique(dt_train$z_class)), verbose=verbose)
  pr_y <- mixtCompPredict(dt_test, model=mcm, algo=mc_model$algo, resLearn=mc_model, nClass=length(unique(dt_train$z_class)), verbose=verbose)$variable$data$z_class$completed
  
  results <- summarise_mc_experiment_results(mc_model, ky_test, pr_y)
  inspect_mc_experiment_results(results)
  
  return(results)
}


save_mc_results <- function(save_dir, results)
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
