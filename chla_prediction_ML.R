library(dplyr)
library(magrittr)
library(corrplot)
library(caret)
library(Hmisc)
library(randomForest)
library(lattice)
library(fmsb)
library(GGally)
library(ggradar)
library(rattle)
library(ggsci)
library(pdp)

data <- read.csv("dataset_chla_ceara.csv") #%>% 
  #filter(!month %in% c(1:6)) %>% 

y <- data %>% 
  dplyr::select(chla, month, year)

x <- data %>% 
  dplyr::select(-c(runoff, temperature_2m, chla))

# Correlation plot -------------------------------------------------------------

mat_cor <- cor(x %>% select(-c(starts_with("acude"), month, year)))
corrplot::corrplot(mat_cor, method = "square", type = "lower", 
                   diag = FALSE, tl.col = "black")

# Train and test splitting -----------------------------------------------------

ind_train <- sample(1:nrow(x), floor(0.8*nrow(x)))
ind_test <- -ind_train

x_train <- x %>%
  slice(ind_train) %>% 
  select(-c(year, month)) %>% 
  mutate(across(.cols = everything(), ~scales::rescale(.x, to = c(0, 1))))

x_test <- x %>%
  slice(ind_test) %>% 
  select(-c(year, month)) %>% 
  mutate(across(.cols = everything(), ~scales::rescale(.x, to = c(0, 1))))

y_train <- y %>%
  slice(ind_train) %>% 
  dplyr::select(chla) %>% 
  as.matrix() %>% 
  as.numeric()

y_test <- y %>%
  slice(ind_test) %>% 
  dplyr::select(chla) %>% 
  as.matrix() %>% 
  as.numeric()

# Model fitting ----------------------------------------------------------------

train_control <- trainControl(method = "cv",
                              number = 5, 
                              returnResamp = "all",
                              savePredictions = "all")

tuneLength_num <- 6

# GLM model
glmnet.mod <- train(x = x_train, y = y_train,
                    method = "glmnet",
                    family = "gaussian",
                    metric = "RMSE",
                    trControl = train_control,
                    tuneLength = tuneLength_num)

# Linear model
lm.mod <- train(x = x_train, y = y_train,
                metric = "RMSE",
                method = "lm",
                trControl = train_control,
                tuneLength = tuneLength_num)

# SVM model
SVMgrid <- expand.grid(sigma = c(10^-4, 10^-3, 10^-2, 10^-1), 
                       C = c(0.25, 0.50, 1.00, 2.00, 4.00, 8.00))

svm_rad.mod <- train(x = x_train, y = y_train,
                     metric = "RMSE",
                     method = "svmRadial",
                     trControl = train_control,
                     tuneGrid = SVMgrid,
                     tuneLength = tuneLength_num)

# Regression tree model
rpart.mod <- train(x = x_train, y = y_train,
                   metric = "RMSE",
                   method = "rpart",
                   trControl = train_control,
                   tuneLength = tuneLength_num)

# RF model
customRF <- list(type = "Regression",
                 library = "randomForest",
                 loop = NULL)

customRF$parameters <- data.frame(parameter = c("mtry", "ntree"),
                                  class = rep("numeric", 2),
                                  label = c("mtry", "ntree"))

customRF$grid <- function(x, y, len = NULL, search = "grid") {}

customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs) {
  randomForest(x, y,
               mtry = param$mtry,
               ntree = param$ntree)
}

#Predict label
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)

#Predict prob
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")

customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

tunegrid <- expand.grid(.mtry = c(2,4,6,8,10,12), 
                        .ntree = c(50,100,250,300))

set.seed(123)
custom <- train(x = x_train, y = y_train,
                metric = "RMSE", 
                method = customRF, 
                tuneGrid = tunegrid, 
                trControl = train_control)

rf.mod <- custom$finalModel

# GBM
gbm.mod <- train(x = x_train, y = y_train,
                 metric = "RMSE",
                 method = "gbm",
                 distribution = "gaussian",
                 trControl = train_control,
                 tuneLength = tuneLength_num
)

# k-NN
knn.mod <- train(x = x_train, y = y_train,
                 method = "knn",
                 metric = "RMSE",
                 trControl = train_control,
                 tuneLength = tuneLength_num
)

# Neural network
nnet_grid <- expand.grid(.decay = c(0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7),                          
                         .size = c(3, 5, 10, 20))
mlp.mod <- train(x = x_train, y = y_train,
                 method = "nnet",
                 metric = "RMSE",
                 trControl = train_control,
                 tuneLength = tuneLength_num, 
                 tuneGrid = nnet_grid,
                 linout = TRUE
)

# Performance evaluation -------------------------------------------------------

predictions <- data.frame(observed = y_test,
                          lm = predict(lm.mod, x_test),
                          glm = predict(glmnet.mod, x_test),
                          knn = predict(knn.mod, x_test),
                          mlp = predict(mlp.mod, x_test),
                          svm_radial = predict(svm_rad.mod, x_test),
                          rtree = predict(rpart.mod, x_test),
                          gbm = predict(gbm.mod, x_test),
                          rf = predict(rf.mod, x_test)) %>% 
  tidyr::pivot_longer(!observed, names_to = "model", values_to = "pred") %>% 
  mutate(model = factor(model, levels = c("lm", "glm", "gam", "knn", 
                                          "svm_linear", "svm_radial", 
                                          "rtree", "gbm", "rf", "mlp"),
                        labels = c("Linear regression", "GLM", "GAM", "KNN",
                                   "SVM - linear kernel", "SVM - radial kernel", 
                                   "Regression Tree", "Gradient Boosting", 
                                   "Random Forest", "MLP")))

# Scatterplot model fit --------------------------------------------------------

ggplot(predictions, aes(x = observed, y = pred)) +
  geom_point(color = "steelblue") +
  geom_abline(slope = 1, intercept = 0) +
  xlim(c(10,80)) +
  ylim(c(10,80)) +
  xlab("Observed") +
  ylab("Predicted") +
  theme_bw() +
  theme(
    strip.background = element_rect(
      color = "black", fill = "#FFFFFF", size = 1.5, linetype = "blank"
    )
  ) +
  facet_wrap(~model)

# Variable importance evaluation -----------------------------------------------

names_models <- c("glmnet" = "GLM",
                  "knn" = "KNN",
                  "nnet" = "MLP",
                  "rpart" = "Reg. Tree",
                  "svmRadial" = "SVM",
                  "gbm" = "GBM",
                  "rf" = "RF")

names_var <- c("water_level" = "Water \nlevel", 
               "precipitation" = "Mean \nprecipitation", 
               "temperature_mean" = "Mean \ntemperature", 
               "volume" = "Volume",
               "radiation" = "Surf. solar \nradiation", 
               "winv" = "Wind speed", 
               "depth" = "Mix-layer \ndepth", 
               "temperature_bottom" = "Bottom \ntemperature",
               "drought_year" = "Drought year",
               "acude_Banabuiú" = "Banabuiú",
               "acude_Castanhão" = "Castanhão",
               "acude_Orós" = "Orós")

extract_imp <- function(mod){
  imp_list <- varImp(mod, scale = TRUE)
  imp <- imp_list$importance %>% 
    data.frame() %>% 
    tibble::rownames_to_column() %>% 
    rename(var = rowname) %>% 
    mutate(model = mod$method) %>% 
    mutate(Overall = scales::rescale(Overall, to = c(0, 1)))
  return(imp)
}

gbm_imp <- data.frame(Overall = gbm::relative.influence(gbm.mod$finalModel, n.trees = gbm.mod$bestTune$n.trees)) %>% 
  tibble::rownames_to_column() %>% 
  rename(var = rowname) %>% 
  mutate(model = "gbm") %>% 
  mutate(Overall = scales::rescale(Overall, to = c(0, 1)))

rf_imp <- data.frame(varImp(rf.mod, scale = TRUE)) %>% 
  tibble::rownames_to_column() %>% 
  rename(var = rowname) %>% 
  mutate(model = "rf") %>% 
  mutate(Overall = scales::rescale(Overall, to = c(0, 1)))

importance_ml <- list(mlp = mlp.mod, rpart = rpart.mod, glm = glmnet.mod) %>% 
  purrr::map_df(~extract_imp(.)) %>% 
  bind_rows(gbm_imp) %>% 
  bind_rows(rf_imp) %>% 
  mutate(across(.cols = c(model), ~stringr::str_replace_all(.x, names_models))) %>% 
  mutate(across(.cols = c(var), ~stringr::str_replace_all(.x, names_var)))

# Plot variable importance -----------------------------------------------------

importance_ml %>% 
  ggplot() +
  geom_boxplot(mapping = aes(x = reorder(var, -Overall, FUN = median), y = Overall), 
               color = "dodgerblue3") + 
  xlab(NULL) +
  ylab("Relative importance") +
  theme_classic() +
  theme(text = element_text(size = 14))

# Plot radar chart -------------------------------------------------------------

teia_dados <- importance_ml %>% 
  tidyr::pivot_wider(names_from = var, values_from = Overall) %>% 
  tidyr::drop_na() %>% 
  tibble::column_to_rownames(var = "model")

teia_dados <- rbind(rep(1, ncol(y)), rep(0, ncol(y)), teia_dados)

pal <- ggsci::pal_jco()
colors_border <- pal(7)
colors_in <- pal(7)

par(mfrow = c(1,1), mar=c(0,0,0.7,0,0))
radarchart(teia_dados,
           axistype = 1,
           title = "Dry season",
           pcol = colors_border, plwd = 2, plty = 1,
           cglcol = "grey", cglty = 1, axislabcol = "grey", 
           caxislabels = seq(0,1,0.25), cglwd = 0.8,
           vlcex = 1.4
)
legend(x=1.7, y=0.5, legend = rownames(teia_dados[-c(1,2),]), bty = "n", pch=20, 
       col=colors_in , text.col = "black", cex=1.4, pt.cex=3)

# Partial dependence plots -----------------------------------------------------

var <- c("water_level", "precipitation", "temperature_mean", "volume",
         "radiation", "winv", "depth", "temperature_bottom")
label_var <- c("Water level (m)", "Mean precipitation (mm)", "Mean temperature (°C)", "Volume (10e6 hm³)",
               "Surf. solar radiation (10e5 J/m²)", "Wind speed (m/s)", "Mix-layer depth (m)", 
               "Bottom temperature (K)")

x_or <- x %>%
  slice(ind_test) %>% 
  select(-c(year, month))

pdp_plot <- list()

fun_transform <- function(x, variavel){
  return(round(x*(max(x_or[variavel]) - min(x_or[variavel])) + min(x_or[variavel]), 2))
}

df_pdp <- data.frame()
for(i in 1:8){
  tmp <- rf.mod %>%
    partial(pred.var = var[i], rug = TRUE, plot.engine = "ggplot2",
            train = x_train) %>%
    data.frame() %>% 
    mutate(across(1, ~fun_transform(.x, var[i]))) %>% 
    mutate(var = label_var[i]) %>% 
    rename(value = 1)
  df_pdp <- rbind(df_pdp, tmp)
}

df_pdp %>% 
  mutate(value = case_when(var == "Surf. solar radiation (10e5 J/m²)" ~ value/100000,
                           TRUE ~ value)) %>% 
  mutate(value = case_when(var == "Volume (10e6 hm³)" ~ value/1000000,
                           TRUE ~ value)) %>% 
  ggplot(aes(value, yhat)) +
  geom_line() +
  geom_smooth(span = 0.8, se = FALSE) +
  xlab(NULL) +
  ylab("Chl a concentration (µg/L)") +
  facet_wrap(~var, scales = "free_x") +
  theme_light() +
  theme(strip.background = element_rect(fill = "white"),
        strip.text = element_text(colour = 'black', size = 12),
        axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10)) 
