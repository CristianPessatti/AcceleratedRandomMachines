require(dplyr)

source("functions/utils/nearest_enemy_distance.R")
source("functions/math/math_formulas.R")

nearest_enemy_sampling <- function(data, y_var, final_sample_size, alpha = 1) {
  data$nearest_enemy_distance <- nearest_enemy_distance(data, y_var) * (-1)

  sampling_weight <- scale_exponential(
    data$nearest_enemy_distance,
    alpha = alpha
  )

  data %>%
    sample_n(final_sample_size, weight = sampling_weight)
}