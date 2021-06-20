# 
# qda.R
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2018-2021 Matteo Spallanzani
# 
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# 

library(MASS)


prepare_qda_data <- function(dt_train, dt_test)
{
  x_train <- dt_train[, !c('k', 'y')]
  y_train <- dt_train$y
  
  x_test <- dt_test[, !c('k', 'y')]
  ky_test <- dt_test[, c('k', 'y')]
  
  ready_data <- list('x_train'=x_train, 'y_train'=y_train, 'x_test'=x_test, 'ky_test'=ky_test)
  return(ready_data)
}


summarise_qda_experiment_results <- function(model, ky_test, pr_y)
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


inspect_qda_experiment_results <- function(results)
{
  cat(sprintf("\nQUADRATIC DISCRIMINANT ANALYSIS\n\n"))
  
  cat(sprintf("Vector-level performance\n"))
  show_accuracy(as.factor(results$model$lev), results$cm_vec)
  
  cat(sprintf("Bag-level performance\n"))
  show_accuracy(as.factor(results$model$lev), results$cm)
}


run_qda_experiment <- function(dt_train, dt_test)
{
  ready_data <- prepare_qda_data(dt_train, dt_test)
  x_train <- ready_data$x_train
  y_train <- ready_data$y_train
  x_test <- ready_data$x_test
  ky_test <- ready_data$ky_test
  
  qda_model <- qda(x_train, y_train)
  pr_y <- predict(qda_model, x_test)$class
  
  results <- summarise_qda_experiment_results(qda_model, ky_test, pr_y)
  inspect_qda_experiment_results(results)
  
  return(results)
}


save_qda_results <- function(save_dir, results)
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
