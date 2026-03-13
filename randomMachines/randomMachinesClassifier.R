require(dplyr)
require(purrr)

source("functions/valdation/balanced_kfold.R")
source("functions/metrics/metrics.R")
source("functions/math/math_formulas.R")
source("functions/valdation/bootstrap_sampler.R")

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
  attr(ensemble_model, "class") <- "randomMachinesClassifierClass"
  return(ensemble_model)
}

randomMachinesClassifierClass <- setClass("randomMachinesClassifierClass",
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

predict.randomMachinesClassifierClass <- function(rmc_model, newdata, type = "class") {
  weights <- rmc_model$bs_weights
  weights <- weights / sum(weights)

  probabilities <- purrr::map(rmc_model$bs_models, function(model) {
    as.matrix(kernlab::predict(model, newdata, type = "probabilities"))
  })

  class_labels <- colnames(probabilities[[1]])

  probabilities <- purrr::map(probabilities, function(mat) {
    mat <- mat[, class_labels, drop = FALSE]
    return(mat)
  })

  weighted_probs <- Reduce(
    `+`,
    purrr::map(seq_along(probabilities), function(i) {
      probabilities[[i]] * weights[i]
    })
  )

  colnames(weighted_probs) <- class_labels

  if (type == "class") {
    max_indices <- apply(weighted_probs, 1, which.max)
    final_preds <- factor(class_labels[max_indices], levels = class_labels)
    return(final_preds)
  } else {
    return(weighted_probs)
  }
}