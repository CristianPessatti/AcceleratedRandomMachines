require(dplyr)

source("functions/utils/nearest_enemy_distance.R")
source("functions/math/math_formulas.R")

#-------------------------------------------------------------------------------
# nearest_enemy_sampling
#
# Sample rows with probability proportional to an exponential transform of the
# negative nearest-enemy distance. Smaller distances (closer to other-class
# points) receive larger weights after scaling.
# - data: data.frame with features and y
# - y_var: name of class column
# - final_sample_size: number of rows to sample
# - alpha: temperature/slope for the exponential scaling
#-------------------------------------------------------------------------------
nearest_enemy_sampling <- function(data, y_var, final_sample_size, alpha = 1) {
  data$nearest_enemy_distance <- nearest_enemy_distance(data, y_var) * (-1)

  sampling_weight <- scale_exponential(
    data$nearest_enemy_distance,
    alpha = alpha
  )

  data %>%
    sample_n(final_sample_size, weight = sampling_weight)
}
