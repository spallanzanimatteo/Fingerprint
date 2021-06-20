# 
# lda.R
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


lda_main <- function(data_set, data_type, save_exp)
{
  dts <- load_data(data_set, data_type)
  dt_train <- dts$train
  dt_test <- dts$test
  
  results <- tryCatch(
    run_lda_experiment(dt_train, dt_test),
    error = function(cond)
    {
      cat(paste0("Error: ", cond))
      return(NULL)
    }
  )
  
  if (save_exp & !is.null(results))
  {
    save_dir <- file.path(get_exp_dir(data_set, data_type), 'LDA')
    save_lda_results(save_dir, results)
  }
}


if (!interactive())
{
  args <- read_cli()
  lda_main(args$data_set, args$data_type, args$save)
}
