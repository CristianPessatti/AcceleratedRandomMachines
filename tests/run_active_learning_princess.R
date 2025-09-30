require(dplyr)
require(ggplot2)
require(arrow)

source("activeLearning/active_learning.R")
source("functions/metrics/metrics.R")
source("functions/sampling/validation_samplers.R")

set.seed(123)

# Load dataset
df <- arrow::read_parquet("datasets/cure_the_princess.parquet")
nrow(df)
df$y <- as.factor(df$y)

# Train/validation split (70/30) stratified
split <- stratified_holdout(df$y, train_prop = 0.7, seed = 123)
train_df <- df[split$train, , drop = FALSE]
valid_df <- df[split$test, , drop = FALSE]

# Run active learning
res <- activeLearning(
  train_df = train_df,
  valid_df = valid_df,
  initial_n = 10L,
  kernel = "rbfdot",
  alpha = 3,
  heterogeneous_prop = 0.8,
  stopping_patience = 10,
  metric_fn = accuracy
)

full_model <- kernlab::ksvm(y ~ ., data = train_df, kernel = "rbfdot")
preds <- predict(full_model, valid_df)
accuracy(preds, valid_df$y)

tail(res$history)
View(res$history)
conv <- res$gganimate_data$convergence

p <- ggplot(conv, aes(iteration, mean_accuracy)) +
  geom_line(color = "steelblue") +
  geom_point(color = "steelblue") +
  geom_line(aes(y = tree_pred), color = "darkorange") +
  ylim(0, 1) +
  labs(title = "Active Learning Convergence",
       x = "Iteration", y = "Mean Accuracy (validation)") +
  theme_minimal() +
  theme(
    title = element_text(size = 24),
    axis.title = element_text(size = 28),
    axis.text = element_text(size = 24)
  ) +
  transition_reveal(iteration)

anim <- animate(p, nframes = nrow(conv), fps = 4)
anim_save("active_learning_convergence.gif", anim)

sc <- res$gganimate_data$scatter

p_sc <- ggplot(sc) +
  geom_point(aes(x, y), color = "gray50", alpha = 0.08) +
  geom_point(
    aes(x, y, color = y_cls,
        size = ifelse(is_just_added, 6, ifelse(is_active, 2, NA))),
    alpha = 1, show.legend = TRUE, na.rm = TRUE
  ) +
  scale_size_identity() +
  labs(title = "Active Sample Highlighting — Iteration: {closest_state}",
       x = "Dim 1", y = "Dim 2", color = "Class") +
  theme_minimal() +
  theme(
    title = element_text(size = 24),
    axis.title = element_text(size = 28),
    axis.text = element_text(size = 24)
  ) +
  transition_states(iteration, transition_length = 0, state_length = 1)

anim_sc <- animate(p_sc, nframes = length(unique(sc$iteration)), fps = 4)
anim_save("active_learning_scatter.gif", anim_sc)