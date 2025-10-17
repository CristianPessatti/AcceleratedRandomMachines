# Load all four classifier implementations
source("randomMachines/randomMachinesClassifier.R")
source("randomMachines/randomMachinesClassifierNE_prev.R")
source("randomMachines/randomMachinesClassifierNE_after.R")
source("randomMachines/randomMachinesClassifierLS_prev.R")

library(dplyr)
library(purrr)
library(ggplot2)

set.seed(42)

# Generate the data
data <- mlbench::mlbench.circle(n = 1000) %>%
  as.data.frame() %>%
  rename(y = classes, x1 = x.1, x2 = x.2)

# Load and assign classifier functions manually
source("randomMachines/randomMachinesClassifier.R")
RM <- randomMachinesClassifier

source("randomMachines/randomMachinesClassifierNE_prev.R")
RM_NE_prev <- randomMachinesClassifier

source("randomMachines/randomMachinesClassifierNE_after.R")
RM_NE_after <- randomMachinesClassifier

source("randomMachines/randomMachinesClassifierLS_prev.R")
RM_LS_prev <- randomMachinesClassifier

clf_list <- list(
  RM = RM,
  RM_NE_prev = RM_NE_prev,
  RM_NE_after = RM_NE_after,
  RM_LS_prev = RM_LS_prev
)

clf_filenames <- c("RM" = "randomMachinesClassifier.R", 
                   "RM_NE_prev" = "randomMachinesClassifierNE_prev.R", 
                   "RM_NE_after" = "randomMachinesClassifierNE_after.R", 
                   "RM_LS_prev" = "randomMachinesClassifierLS_prev.R")

# Helper function to perform a single holdout
do_holdout <- function(clf_fun, data, B = 50) {
  n <- nrow(data)
  test_idx <- sample(seq_len(n), floor(0.3 * n))
  train_idx <- setdiff(seq_len(n), test_idx)
  train <- data[train_idx, ]
  test  <- data[test_idx, ]
  
  # Try-catch to prevent the workflow from breaking
  res <- tryCatch({
    t0 <- Sys.time()
    model <- clf_fun(y ~ x1 + x2, data = train, B = B)
    elapsed <- as.numeric(Sys.time() - t0, units = "secs")
    preds <- predict(model, test, type = "class")
    acc <- accuracy(test$y, preds)
    et <- if (!is.null(model$elapsed_time)) model$elapsed_time else elapsed
    list(acc = acc, time = et)
  }, error = function(e) {
    list(acc = NA, time = NA)
  })
  return(res)
}

# Run 10 holdouts for each classifier
all_results <- list()
n_runs <- 10

for (clf_key in names(clf_list)) {
  clf_fun <- clf_list[[clf_key]]
  res <- map(1:n_runs, ~do_holdout(clf_fun, data))
  accs <- map_dbl(res, "acc")
  times <- map_dbl(res, "time")
  all_results[[clf_key]] <- data.frame(
    method = clf_key,
    accuracy = accs,
    elapsed_time = times
  )
}

results_df <- bind_rows(all_results)

# Boxplot of accuracies
ggplot(results_df, aes(x = method, y = accuracy, fill = method)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Random Machines Classifier Accuracies (10 Holdouts)", y = "Accuracy", x = "Method")

# Boxplot of elapsed times
ggplot(results_df, aes(x = method, y = elapsed_time, fill = method)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Random Machines Classifier Elapsed Times (10 Holdouts)", y = "Elapsed time (seconds)", x = "Method")
