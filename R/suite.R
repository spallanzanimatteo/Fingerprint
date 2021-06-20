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
