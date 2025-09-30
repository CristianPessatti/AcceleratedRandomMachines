require(dplyr)

#-------------------------------------------------------------------------------
# create_partitions
#
# Map numeric features to an integer partition id by cutting each numeric
# feature into n_partitions bins and encoding the multi-dimensional bin as a
# single index. Non-numeric columns are ignored.
#-------------------------------------------------------------------------------
create_partitions <- function(df, n_partitions = 4) {
  cuts <- df %>%
    mutate(across(where(is.numeric),
                  ~ cut(.x,
                        breaks = seq(min(.x, na.rm = TRUE),
                                     max(.x, na.rm = TRUE),
                                     length.out = n_partitions + 1),
                        include.lowest = TRUE,
                        labels = FALSE)))

  groups <- cuts %>% select(where(is.numeric))

  vector_partition <- apply(groups, 1, function(line) {
    sum((line - 1) * n_partitions ^ (seq_along(line) - 1)) + 1
  })

  return(factor(vector_partition))
}
