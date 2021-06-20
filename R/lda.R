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
