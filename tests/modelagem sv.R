require(kernlab)
require(ranger)
require(dplyr)

mod <- ksvm(Species~., 
            data=iris, kernel="rbfdot")




# amostra ativa
amostra.ativa <- iris %>% sample_n(50)

mod <- ksvm(Species~., 
            data=amostra.ativa, kernel="rbfdot")

id.sv <- mod@alphaindex %>% unlist() %>% unique() %>% sort()
Y.sv <- rep("0", nrow(amostra.ativa))
Y.sv[id.sv]="1"
Y.sv <- as.factor(Y.sv)


# modelando os svs da amostra ativa
amostra.ativa.sv <- 
  bind_cols(amostra.ativa,data.frame(Y.sv)) %>% 
  select(-Species)

mod.rf <- ranger(Y.sv ~ ., 
                 data = amostra.ativa.sv, 
                 num.trees = 1,
                 probability = T)




np=10

#prevendo sv da amostra pool
pool <- iris %>%  sample_n(np)

Y.pool <- pool %>% select(Species) 
X.pool <- pool %>% select(-Species) 
res <- mod.rf %>% predict(X.pool,type="response")
prob.sv <- res$predictions[,2]

RES <- data.frame(Y.pool, prob.sv) %>% 
  mutate(id = row_number())


#retirando o 'melhor vetor de suporte' da cada categoria do pool
RES  %>% 
  group_by(Species) %>%
  slice(which.max(prob.sv))
