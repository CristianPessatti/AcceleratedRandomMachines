require(dplyr)
require(ggplot2)
require(arrow)

source("activeLearning/active_learning.R")
source("functions/metrics/metrics.R")
source("functions/sampling/validation_samplers.R")

set.seed(123)

# Load dataset
df <- arrow::read_parquet("datasets/linear_split.parquet")
df$y <- as.factor(df$y)

# Train/validation split (70/30) stratified
split <- stratified_holdout(df$y, train_prop = 0.7, seed = 123)
train_df <- df[split$train, , drop = FALSE]
valid_df <- df[split$test, , drop = FALSE]
# Run active learning
res <- activeLearning(
  train_df = train_df,
  valid_df = valid_df,
  initial_n = 20L,
  kernel = "vanilladot",
  alpha = 5,
  heterogeneous_prop = 0.8,
  stopping_patience = 15,
  metric_fn = accuracy
)

conv <- res$gganimate_data$convergence

library(ggplot2)
library(gganimate)

p <- ggplot(conv, aes(iteration, mean_accuracy)) +
  geom_line(color = "steelblue") +
  geom_point(color = "steelblue") +
  geom_line(aes(y = tree_pred), color = "darkorange") +
  ylim(0, 1) +
  labs(title = "Active Learning Convergence",
       x = "Iteration", y = "Mean Accuracy (validation)") +
  transition_reveal(iteration)

anim <- animate(p, nframes = nrow(conv), fps = 2)
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
  transition_states(iteration, transition_length = 0, state_length = 1)

anim_sc <- animate(p_sc, nframes = length(unique(sc$iteration)), fps = 2)
anim_save("active_learning_scatter.gif", anim_sc)






ddd <- res$history %>% select(iteration, mean_accuracy) #%>%
  #rbind(tibble(iteration = 12:40, mean_accuracy = rep(0.95, 29)))

tree_model <- rpart::rpart(mean_accuracy ~ iteration, 
  data = ddd,
  control = rpart::rpart.control(minsplit = 2))
ddd$preds <- as.numeric(predict(tree_model, ddd))

res$history %>%
  ggplot(aes(x = iteration, y = mean_accuracy)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(color = "steelblue", size = 1.5) +
  geom_line(data = ddd, aes(x = iteration, y = preds), color = "red", linewidth = 1) +
  ylim(0, 1) +
  theme_minimal()

# Plot convergence
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

train_df %>%
  ggplot(aes(x = x1, y = x2, colour = y)) +
  geom_point() +
  theme_minimal()

res$final_active %>%
  ggplot(aes(x = x1, y = x2, colour = y)) +
  geom_point() +
  theme_minimal()

count_sv <- res$support_vector_counts %>%
  group_by(train_row_id) %>%
  summarise(count_sv = sum(sv_count))

sv_df <- res$final_active %>%
  left_join(count_sv, by = c(".row_id" = "train_row_id"))

sv_df %>%
  ggplot(aes(x = x1, y = x2, colour = y, size = count_sv)) +
  geom_point(alpha = 0.5) +
  labs(title = "",
       x = "X1",
       y = "X2",
       size = "Support Vector Count") +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 28),
    axis.text = element_text(size = 24),
    legend.position = "none"
  )

tail(res$history)

modelo_completo <- kernlab::ksvm(y ~ x1 + x2, data = df, kernel = "vanilladot")
preds <- predict(modelo_completo, df)
accuracy(preds, df$y)
