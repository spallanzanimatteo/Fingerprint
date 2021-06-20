# 
# suite.R
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

source('utils.R')
source(file.path('methods', 'lda.R'))
source(file.path('methods', 'qda.R'))
source(file.path('methods', 'mixtcomp.R'))


suite <- function(data_set, data_type, save_exp)
{
  exp_dir <- get_exp_dir(data_set, data_type)
  
  dts <- load_data(data_set, data_type)
  dt_train <- dts$train
  dt_test <- dts$test
  
  # LDA
  lda_results <- tryCatch(
    run_lda_experiment(dt_train, dt_test),
    error = function(cond)
    {
      cat(paste0("Error: ", cond))
      return(NULL)
    }
  )
  if (save_exp & !is.null(lda_results))
  {
    lda_save_dir <- file.path(exp_dir, 'LDA')
    save_lda_results(lda_save_dir, lda_results)
  }
  
  # QDA
  qda_results <- tryCatch(
    run_qda_experiment(dt_train, dt_test),
    error = function(cond)
    {
      cat(paste0("Error: ", cond))
      return(NULL)
    }
  )
  if (save_exp & !is.null(qda_results))
  {
    qda_save_dir <- file.path(exp_dir, 'QDA')
    save_qda_results(qda_save_dir, qda_results)
  }
  
  # MixtComp
  mc_results <- tryCatch(
    run_mc_experiment(dt_train, dt_test),
    error = function(cond)
    {
      cat(paste0("Error: ", cond))
      return(NULL)
    }
  )
  if (save_exp & !is.null(mc_results))
  {
    mc_save_dir <- file.path(get_exp_dir(data_set, data_type), 'MixtComp')
    save_mc_results(mc_save_dir, mc_results)
  }
}


if (!interactive())
{
  args <- read_cli()
  suite(args$data_set, args$data_type, args$save)
}
