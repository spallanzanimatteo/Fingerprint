source('utils.R')
source(file.path('methods', 'qda.R'))


qda_main <- function(data_set, data_type, save_exp)
{
  dts <- load_data(data_set, data_type)
  dt_train <- dts$train
  dt_test <- dts$test
  
  results <- tryCatch(
    run_qda_experiment(dt_train, dt_test),
    error = function(cond)
    {
      cat(paste0("Error: ", cond))
      return(NULL)
    }
  )
  
  if (save_exp & !is.null(results))
  {
    save_dir <- file.path(get_exp_dir(data_set, data_type), 'QDA')
    save_qda_results(save_dir, results)
  }
}


if (!interactive())
{
  args <- read_cli()
  qda_main(args$data_set, args$data_type, args$save)
}
