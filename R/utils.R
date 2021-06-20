# 
# utils.R
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

library(optparse)
library(openxlsx)
library(data.table)


read_cli <- function()
{
  options <- list(
    make_option('--data_set'),
    make_option('--data_type', type='character', help='Numerical-only (`numerical`), categorical-only (`categorical`), mixed (`mixed`)'),
    make_option('--save', action='store_true', default=FALSE, help='Save models and results to disk')
  )
  parser <- OptionParser(option_list=options)
  args <- parse_args(parser)
  
  stopifnot(match(args$data_type, c('numerical', 'categorical', 'mixed'), nomatch=0) > 0)
  
  return(args)
}


get_exp_dir <- function(data_set, data_type)
{
  exp_dir <- file.path('..', 'Data', data_set, 'Experiments', paste0(toupper(substr(data_type, 1, 1)), substr(data_type, 2, nchar(data_type))))
  return(exp_dir)
}


load_data <- function(data_set, data_type)
{
  data_file <- file.path('..', 'Data', data_set, 'data_set.xlsx')
  dt_train <- as.data.table(read.xlsx(data_file, sheet=1))
  dt_test <- as.data.table(read.xlsx(data_file, sheet=2))
  
  if (data_type == 'numerical')
  {
    q_names <- names(dt_train)[startsWith(names(dt_train), 'q')]
    dt_train <- dt_train[, !..q_names]
    dt_test <- dt_test[, !..q_names]
  }
  else if (data_type == 'categorical')
  {
    x_names <- names(dt_train)[startsWith(names(dt_train), 'x')]
    dt_train <- dt_train[, !..x_names]
    dt_test <- dt_test[, !..x_names]
  }
  
  dts <- list('train'=dt_train, 'test'=dt_test)
  return(dts)
}


get_mode <- function(v)
{
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}


show_accuracy <- function(classes, cm)
{
  cat(sprintf("Confusion matrix:"))
  print(cm)

  per_class = diag(cm) / rowSums(cm)
  for (iy in 1:length(classes))
  {
    cat(sprintf("Accuracy on class %2d: %6.2f%%\n", as.integer(classes[[iy]]), 100.0 * as.numeric(per_class[[iy]])))
  }
  
  overall = sum(diag(cm)) / sum(cm)
  cat(sprintf("Overall accuracy:     %6.2f%%\n\n", 100.0 * overall))
}
