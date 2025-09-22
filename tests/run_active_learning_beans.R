require(dplyr)
require(ggplot2)
require(arrow)

source("activeLearning/active_learning.R")
source("functions/sampling/validation_samplers.R")

set.seed(123)

# Load dataset
df <- arrow::read_parquet("datasets/beans.parquet")# %>%
#  filter(y != "BOMBAY")
df$y <- factor(df$y)

df %>%
  group_by(y) %>%
  summarise(n = n())

# Train/validation split (70/30) stratified
split <- stratified_holdout(df$y, train_prop = 0.7, seed = 123)
train_df <- df[split$train, , drop = FALSE]
valid_df <- df[split$test, , drop = FALSE]

nrow(df)

mod0 <- kernlab::ksvm(y ~ ., data = train_df, kernel = "rbfdot", C = 1)
preds0 <- predict(mod0, valid_df)
mean(preds0 == valid_df$y) - max(res$history$mean_accuracy)

# Run active learning
res <- activeLearning(
  train_df = train_df,
  valid_df = valid_df,
  initial_n = 100L,
  kernel = "rbfdot",
  alpha = 3,
  heterogeneous_prop = 0.8,
  stopping_patience = 15,
  max_additions = 1000
)

p <- ggplot(res$history, aes(x = iteration, y = mean_accuracy)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(color = "steelblue", size = 1.5) +
  geom_line(aes(x = iteration, y = plateau_max_pred), color = "red", linewidth = 1) +
  ylim(0, 1) +
  theme_minimal() +
  labs(title = "Active Learning Convergence (Linear Split)",
       x = "Iteration",
       y = "Mean Accuracy (validation)")

p

p1 <-train_df %>%
  ggplot(aes(x = x1, y = x2, colour = y)) +
  geom_point() +
  theme_minimal()


nrow(res$final_active)
p2 <- res$final_active %>%
  ggplot(aes(x = x1, y = x2, colour = y)) +
  geom_point() +
  theme_minimal()

patchwork::wrap_plots(p1, p2, ncol = 2)

count_sv <- res$support_vector_counts %>%
  group_by(train_row_id) %>%
  summarise(count_sv = sum(sv_count))

sv_df <- res$final_active %>%
  left_join(count_sv, by = c(".row_id" = "train_row_id"))

sv_df %>%
  ggplot(aes(x = x1, y = x2, colour = y, size = count_sv)) +
  geom_point(alpha = 0.5) +
  theme_minimal()
