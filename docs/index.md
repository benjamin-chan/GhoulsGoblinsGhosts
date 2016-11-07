---
title: "Kaggle: Ghouls, Goblins, and Ghosts... Boo!"
date: "2016-11-07 09:59:10"
author: Benjamin Chan (benjamin.ks.chan@gmail.com)
output:
  html_document:
    toc: true
    theme: simplex
---

---

# Preamble

Set working directory.


```r
setwd("~/Projects/Kaggle/GhoulsGoblinsGhosts/scripts")
```

Load libraries.


```r
library(plyr)
library(dplyr)
library(reshape2)
library(ggplot2)
library(caret)
library(parallel)
library(doParallel)
```

Reproducibility steps.


```r
sessionInfo()
```

```
## R version 3.3.1 (2016-06-21)
## Platform: x86_64-w64-mingw32/x64 (64-bit)
## Running under: Windows 7 x64 (build 7601) Service Pack 1
## 
## locale:
## [1] LC_COLLATE=English_United States.1252 
## [2] LC_CTYPE=English_United States.1252   
## [3] LC_MONETARY=English_United States.1252
## [4] LC_NUMERIC=C                          
## [5] LC_TIME=English_United States.1252    
## 
## attached base packages:
## [1] parallel  stats     graphics  grDevices utils     datasets  methods  
## [8] base     
## 
## other attached packages:
##  [1] xgboost_0.4-4       doParallel_1.0.10   iterators_1.0.8    
##  [4] foreach_1.4.3       caret_6.0-71        lattice_0.20-34    
##  [7] ggplot2_2.1.0       reshape2_1.4.1      dplyr_0.5.0        
## [10] plyr_1.8.4          rmarkdown_1.0       knitr_1.14         
## [13] checkpoint_0.3.16   RevoUtilsMath_8.0.3
## 
## loaded via a namespace (and not attached):
##  [1] Rcpp_0.12.7        compiler_3.3.1     formatR_1.4       
##  [4] nloptr_1.0.4       class_7.3-14       tools_3.3.1       
##  [7] digest_0.6.10      lme4_1.1-12        evaluate_0.9      
## [10] tibble_1.2         gtable_0.2.0       nlme_3.1-128      
## [13] mgcv_1.8-15        Matrix_1.2-7.1     DBI_0.5-1         
## [16] SparseM_1.72       e1071_1.6-7        stringr_1.1.0     
## [19] MatrixModels_0.4-1 RevoUtils_10.0.1   stats4_3.3.1      
## [22] grid_3.3.1         nnet_7.3-12        data.table_1.9.6  
## [25] R6_2.2.0           minqa_1.2.4        car_2.1-3         
## [28] magrittr_1.5       scales_0.4.0       codetools_0.2-15  
## [31] htmltools_0.3.5    MASS_7.3-45        splines_3.3.1     
## [34] assertthat_0.1     pbkrtest_0.4-6     colorspace_1.2-7  
## [37] labeling_0.3       quantreg_5.29      stringi_1.1.1     
## [40] lazyeval_0.2.0     munsell_0.4.3      chron_2.3-47
```

```r
set.seed(as.integer(as.Date("2016-11-04")))
```

Source user-defined functions.


```r
sapply(list.files("../lib", full.names = TRUE), source)
```

```
## named list()
```

---

# Read data

Read the data stored locally.


```r
unzip("../data/raw/train.csv.zip", exdir = tempdir())
train <- read.csv(file.path(tempdir(), "train.csv"), stringsAsFactors = TRUE)
unzip("../data/raw/test.csv.zip", exdir = tempdir())
test <- read.csv(file.path(tempdir(), "test.csv"), stringsAsFactors = TRUE)
```

List the columns in both data sets.


```r
merge(data.frame(col = names(train), inTrain = TRUE),
      data.frame(col = names(test), inTest = TRUE),
      by = "col", all = TRUE) %>%
  filter(inTrain & inTest) %>%
  .[, "col"]
```

```
## [1] bone_length   color         hair_length   has_soul      id           
## [6] rotting_flesh
## 7 Levels: bone_length color hair_length has_soul id ... type
```

Check that the columns in `train` are the same as in `test`.
Show the columns that are not in both data sets.


```r
merge(data.frame(col = names(train), inTrain = TRUE),
      data.frame(col = names(test), inTest = TRUE),
      by = "col", all = TRUE) %>%
  filter(is.na(inTrain) | is.na(inTest))
```

```
##    col inTrain inTest
## 1 type    TRUE     NA
```

---

# Explore sample

Describe the `type` variable.


```r
tab <- table(train$type)
data.frame(cbind(freq = tab, prop = prop.table(tab)))
```

```
##        freq      prop
## Ghost   117 0.3153639
## Ghoul   129 0.3477089
## Goblin  125 0.3369272
```

Plot densities between `type` and the numeric variables.
Look for patterns or clusterings.


```r
numvar <- names(train)[sapply(train, class) == "numeric"]
melt(train, id.vars = "type", measure.vars = numvar) %>%
  ggplot(aes(x = type, y = value, group = type, color = type, fill = type)) +
    geom_violin(alpha = 1/2) +
    facet_wrap(~ variable) +
    theme_bw() +
    theme(legend.position = "none")
```

![plot of chunk densities](../figures/densities-1.png)

Show summary statistics for numeric variables.


```r
select(train, matches(paste(numvar, collapse = "|"))) %>% summary()
```

```
##   bone_length      rotting_flesh      hair_length        has_soul       
##  Min.   :0.06103   Min.   :0.09569   Min.   :0.1346   Min.   :0.009402  
##  1st Qu.:0.34001   1st Qu.:0.41481   1st Qu.:0.4074   1st Qu.:0.348002  
##  Median :0.43489   Median :0.50155   Median :0.5386   Median :0.466372  
##  Mean   :0.43416   Mean   :0.50685   Mean   :0.5291   Mean   :0.471392  
##  3rd Qu.:0.51722   3rd Qu.:0.60398   3rd Qu.:0.6472   3rd Qu.:0.600610  
##  Max.   :0.81700   Max.   :0.93247   Max.   :1.0000   Max.   :0.935721
```

Show the standard deviations of the numeric variables.


```r
select(train, matches(paste(numvar, collapse = "|"))) %>% var() %>% diag() %>% sqrt()
```

```
##   bone_length rotting_flesh   hair_length      has_soul 
##     0.1328331     0.1463577     0.1699018     0.1761293
```

The numeric variables are pre-scaled with values between 0.00940161587866194, 1.
Therefore, no preprocessing is needed.

Plot bivariate densities between numeric variables.
Look for correlations to reduce dimensionality of data.


```r
corr <-
  select(train, matches(paste(numvar, collapse = "|"))) %>%
  cor()
contHighCorr <- colnames(corr)[findCorrelation(corr)]
sprintf("Remove variable due to high pair-wise correlation with other variables: %s",
        contHighCorr)
```

```
## character(0)
```

```r
replace(corr, which(upper.tri(corr, diag = TRUE)), NA) %>%
  melt(na.rm = TRUE) %>%
  ggplot(aes(x = Var1, y = Var2, fill = value)) + 
    geom_tile(color = "white") +
    scale_fill_gradient2(low = "blue",
                         high = "red",
                         mid = "white",
                         midpoint = 0,
                         limit = c(-1, 1),
                         space = "Lab",
                         name="R") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          panel.grid.major = element_blank()) +
    coord_fixed()
```

![plot of chunk heatmapCorr](../figures/heatmapCorr-1.png)

```r
round(corr, 2)
```

```
##               bone_length rotting_flesh hair_length has_soul
## bone_length          1.00         -0.04        0.35     0.38
## rotting_flesh       -0.04          1.00       -0.22    -0.13
## hair_length          0.35         -0.22        1.00     0.47
## has_soul             0.38         -0.13        0.47     1.00
```

Examine association between `color` and `type`.


```r
table(train$color)
```

```
## 
## black blood  blue clear green white 
##    41    12    19   120    42   137
```

```r
table(train$color, train$type) %>%
  prop.table(margin = 1)
```

```
##        
##             Ghost     Ghoul    Goblin
##   black 0.3414634 0.3414634 0.3170732
##   blood 0.5000000 0.3333333 0.1666667
##   blue  0.3157895 0.3157895 0.3684211
##   clear 0.2666667 0.3500000 0.3833333
##   green 0.3571429 0.3095238 0.3333333
##   white 0.3211679 0.3649635 0.3138686
```

---

# Model on `train`

Set the control parameters.


```r
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 10,
                     classProbs=TRUE,
                     savePredictions = TRUE,
                     allowParallel = FALSE,
                     search = "random")
```

Set the model.


```r
library(xgboost)
method <- "xgbTree"
```

Set the tuning grid for model xgbTree.


```r
grid <- expand.grid(mtry = seq(5, 25, 5))
```

Fit model over the tuning parameters.


```r
trainingModel <- train(type ~ .,
                       data = select(train, -matches("id")),
                       method = method,
                       trControl = ctrl,
                       # tuneGrid = grid,
                       tuneLength = 10,
                       nthreads = 3)
```

Evaluate the model on the training dataset.


```r
trainingModel
```

```
## eXtreme Gradient Boosting 
## 
## 371 samples
##   5 predictor
##   3 classes: 'Ghost', 'Ghoul', 'Goblin' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold, repeated 10 times) 
## Summary of sample sizes: 336, 333, 334, 334, 333, 334, ... 
## Resampling results across tuning parameters:
## 
##   eta         max_depth  gamma      colsample_bytree  min_child_weight
##   0.03417152  7          5.7063216  0.3442821          0              
##   0.08739976  7          6.9796986  0.3454693         20              
##   0.15129156  6          9.6279246  0.5388811         11              
##   0.22148854  7          9.4140174  0.5047818          4              
##   0.32696169  6          0.3385011  0.3736250          4              
##   0.41947226  1          2.2502930  0.4784351          2              
##   0.45269945  1          0.8618729  0.6690997         13              
##   0.45740450  4          7.8663852  0.5361600          6              
##   0.52993226  6          6.5322837  0.5669058          4              
##   0.59592105  9          8.8393230  0.4287785         12              
##   nrounds  Accuracy   Kappa    
##   542      0.7033460  0.5546715
##   326      0.6993318  0.5486208
##   995      0.6858649  0.5284797
##   942      0.6884798  0.5325485
##   818      0.6953917  0.5424245
##   358      0.7254858  0.5878549
##   547      0.7224574  0.5833094
##   353      0.6955334  0.5429394
##     3      0.6848569  0.5268844
##   573      0.7001221  0.5497784
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were nrounds = 358, max_depth = 1,
##  eta = 0.4194723, gamma = 2.250293, colsample_bytree = 0.4784351
##  and min_child_weight = 2.
```

```r
ggplot(trainingModel) + theme_bw()
```

![plot of chunk tuningTraining](../figures/tuningTraining-1.png)

```r
hat <-
  train %>%
  transform(hat = predict(trainingModel, train)) %>%
  select(matches("type|hat"))
postResample(hat$hat, hat$type)
```

```
##  Accuracy     Kappa 
## 0.8032345 0.7046629
```

```r
confusionMatrix(hat$hat, hat$type)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction Ghost Ghoul Goblin
##     Ghost    106     3     10
##     Ghoul      0   103     26
##     Goblin    11    23     89
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8032          
##                  95% CI : (0.7591, 0.8425)
##     No Information Rate : 0.3477          
##     P-Value [Acc > NIR] : <2e-16          
##                                           
##                   Kappa : 0.7047          
##  Mcnemar's Test P-Value : 0.3573          
## 
## Statistics by Class:
## 
##                      Class: Ghost Class: Ghoul Class: Goblin
## Sensitivity                0.9060       0.7984        0.7120
## Specificity                0.9488       0.8926        0.8618
## Pos Pred Value             0.8908       0.7984        0.7236
## Neg Pred Value             0.9563       0.8926        0.8548
## Prevalence                 0.3154       0.3477        0.3369
## Detection Rate             0.2857       0.2776        0.2399
## Detection Prevalence       0.3208       0.3477        0.3315
## Balanced Accuracy          0.9274       0.8455        0.7869
```

```r
varImp(trainingModel)
```

```
## xgbTree variable importance
## 
##               Overall
## hair_length   100.000
## has_soul       94.692
## bone_length     6.175
## rotting_flesh   0.000
```

Display the final model.


```r
trainingModel$finalModel
```

---

# Predict on `test`

Apply the model to the `test` data.


```r
hat <-
  test %>%
  transform(type = predict(trainingModel, test),
            pr = predict(trainingModel, test, type = "prob")) %>%
  select(matches("id|type|pr."))
dim(hat)
```

```
## [1] 529   5
```

```r
str(hat)
```

```
## 'data.frame':	529 obs. of  5 variables:
##  $ id       : int  3 6 9 10 13 14 15 16 17 18 ...
##  $ type     : Factor w/ 3 levels "Ghost","Ghoul",..: 2 3 2 1 1 1 2 2 3 3 ...
##  $ pr.Ghost : num  0.0113 0.1236 0.0259 0.4004 0.9598 ...
##  $ pr.Ghoul : num  0.8024 0.34 0.4969 0.2611 0.0105 ...
##  $ pr.Goblin: num  0.1864 0.5365 0.4772 0.3385 0.0297 ...
```

```r
head(hat)
```

```
##   id   type   pr.Ghost   pr.Ghoul  pr.Goblin
## 1  3  Ghoul 0.01127358 0.80235976 0.18636660
## 2  6 Goblin 0.12359555 0.33995229 0.53645211
## 3  9  Ghoul 0.02588930 0.49693584 0.47717485
## 4 10  Ghost 0.40038383 0.26114765 0.33846849
## 5 13  Ghost 0.95980191 0.01052354 0.02967460
## 6 14  Ghost 0.95279258 0.01272845 0.03447893
```

Describe the `type` variable.


```r
tab <- table(hat$type)
data.frame(cbind(freq = tab, prop = prop.table(tab)))
```

```
##        freq      prop
## Ghost   190 0.3591682
## Ghoul   172 0.3251418
## Goblin  167 0.3156900
```

Save the predictions to file.


```r
options(scipen = 10)
select(hat, matches("id|type")) %>%
  write.csv(file = "../data/processed/submission.csv", row.names = FALSE, quote = FALSE)
file.info("../data/processed/submission.csv")
```

```
##                                  size isdir mode               mtime
## ../data/processed/submission.csv 5942 FALSE  666 2016-11-07 10:32:54
##                                                ctime               atime
## ../data/processed/submission.csv 2016-11-06 15:16:49 2016-11-06 15:16:49
##                                  exe
## ../data/processed/submission.csv  no
```
