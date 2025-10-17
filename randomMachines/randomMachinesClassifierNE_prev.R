require(dplyr)
require(purrr)

source("functions/valdation/balanced_kfold.R")
source("functions/metrics/metrics.R")
source("functions/math/math_formulas.R")
source("functions/valdation/bootstrap_sampler.R")
source("functions/sampling/nearest_enemy_sampling.R")

randomMachinesClassifier <- function(
  formula,
  data,
  B = 50,
  kernels = list(
    kernlab::vanilladot(),
    kernlab::polydot(2),
    kernlab::rbfdot(1),
    kernlab::laplacedot(1)
  ),
  loss_function = accuracy,
  C = 1
) {
  if(class(formula) == "character") {
    form <- as.formula(formula) 
  } else if (class(formula) == "formula") {
    form <- formula
  } else {
    stop("formula must be a character or formula")
  }

  target <- as.character(form)[2]
  
  data <- nearest_enemy_sampling(data, y_var = target, final_sample_size = nrow(data) * 0.5, alpha = 3)

  initial_time <- Sys.time()

  folds <- stratified_kfold(data, K = B, y = target)

  models <- map_dbl(kernels, function(kernel) {
    map_dbl(folds, function(fold) {
      model <- kernlab::ksvm(form, data = data[fold$train, ], kernel = kernel, C = C)
      y_pred <- kernlab::predict(model, data[fold$test, ])
      loss_function(data[[target]][fold$test], y_pred)
    }) %>%
    mean()
  })

  lambda <- log_normalize(models)

  bs_samples <- sample_bootstrap(data, n = B)

  bs_kernels <- sample(kernels, size = B, replace = TRUE, prob = lambda)

  bs_models <- map(1:B, function(i) {
    kernlab::ksvm(form, data = data[bs_samples[[i]]$train, ], kernel = bs_kernels[[i]], C = C, prob.model = TRUE)
  })

  bs_losses <- map_dbl(1:B, function(i) {
    y_pred <- kernlab::predict(bs_models[[i]], data[bs_samples[[i]]$test, ])
    loss_function(data[[target]][bs_samples[[i]]$test], y_pred)
  })

  bs_weights <- log_normalize(bs_losses)

  elapsed_time <- as.numeric(Sys.time() - initial_time, units = "secs")

  ensemble_model <- list(
    train = data,
    target = target,
    kernels = kernels,
    kernel_lambdas = lambda,
    bs_models = bs_models,
    bs_samples = bs_samples,
    bs_weights = bs_weights,
    elapsed_time = elapsed_time
  )
  attr(ensemble_model, "class") <- "randomMachinesClassifierNEClass"
  return(ensemble_model)
}

randomMachinesClassifierNEClass <- setClass("randomMachinesClassifierNEClass",
  slots = list(
    train = "data.frame",
    target = "character",
    kernels = "list",
    kernel_lambdas = "numeric",
    bs_models = "list",
    bs_samples = "list",
    bs_weights = "numeric",
    elapsed_time = "numeric"
  )
)

predict.randomMachinesClassifierNEClass <- function(rmc_model, newdata, type = "class") {
  labels <- rmc_model$train[, rmc_model$target]

  probabilities <- map(1:length(rmc_model$bs_models), function(i) {
    kernlab::predict(rmc_model$bs_models[[i]], newdata, type = "probabilities")
  })

  prob_array <- array(unlist(probabilities), dim = c(nrow(newdata), ncol(probabilities[[1]]), length(rmc_model$bs_models)))
  
  mean_probs <- apply(prob_array, c(1, 2), mean)
  
  class_labels <- colnames(probabilities[[1]])
  colnames(mean_probs) <- class_labels

  if (type == "class") {
    max_indices <- apply(mean_probs, 1, which.max)
    final_preds <- as.factor(class_labels[max_indices])
    return(final_preds)
  } else {
    return(mean_probs)
  }
}