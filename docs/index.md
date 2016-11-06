---
title: "Kaggle: Ghouls, Goblins, and Ghosts... Boo!"
date: "2016-11-06 15:06:21"
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
##  [1] C50_0.1.0-24        doParallel_1.0.10   iterators_1.0.8    
##  [4] foreach_1.4.3       caret_6.0-71        lattice_0.20-34    
##  [7] ggplot2_2.1.0       reshape2_1.4.1      dplyr_0.5.0        
## [10] plyr_1.8.4          rmarkdown_1.0       knitr_1.14         
## [13] checkpoint_0.3.16   RevoUtilsMath_8.0.3
## 
## loaded via a namespace (and not attached):
##  [1] Rcpp_0.12.7        compiler_3.3.1     nloptr_1.0.4      
##  [4] formatR_1.4        class_7.3-14       tools_3.3.1       
##  [7] partykit_1.1-1     digest_0.6.10      lme4_1.1-12       
## [10] evaluate_0.9       tibble_1.2         gtable_0.2.0      
## [13] nlme_3.1-128       mgcv_1.8-15        Matrix_1.2-7.1    
## [16] DBI_0.5-1          SparseM_1.72       e1071_1.6-7       
## [19] stringr_1.1.0      MatrixModels_0.4-1 RevoUtils_10.0.1  
## [22] stats4_3.3.1       grid_3.3.1         nnet_7.3-12       
## [25] R6_2.2.0           survival_2.39-5    Formula_1.2-1     
## [28] minqa_1.2.4        car_2.1-3          magrittr_1.5      
## [31] scales_0.4.0       codetools_0.2-15   htmltools_0.3.5   
## [34] MASS_7.3-45        splines_3.3.1      assertthat_0.1    
## [37] pbkrtest_0.4-6     colorspace_1.2-7   labeling_0.3      
## [40] quantreg_5.29      stringi_1.1.1      lazyeval_0.2.0    
## [43] munsell_0.4.3
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
ctrl <- trainControl(method = "cv",
                     number = 10,
                     classProbs = TRUE,
                     savePredictions = TRUE,
                     allowParallel = TRUE)
```

Set the model.


```r
library(C50)
method <- "C5.0"
```

Set the tuning grid for model C5.0.


```r
grid <- expand.grid(trials = seq(10, 40, 10),
                    model = c("rules", "tree"),
                    winnow = c(TRUE, FALSE))
```

Fit model over the tuning parameters.


```r
cl <- makeCluster(3)
registerDoParallel(cl)
trainingModel <- train(type ~ .,
                       data = select(train, -matches("id")),
                       method = method,
                       trControl = ctrl,
                       tuneGrid = grid)
stopCluster(cl)
```

Evaluate the model on the training dataset.


```r
trainingModel
```

```
## C5.0 
## 
## 371 samples
##   5 predictor
##   3 classes: 'Ghost', 'Ghoul', 'Goblin' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 336, 333, 334, 334, 333, 334, ... 
## Resampling results across tuning parameters:
## 
##   model  winnow  trials  Accuracy   Kappa    
##   rules  FALSE   10      0.7112924  0.5667289
##   rules  FALSE   20      0.7139362  0.5706837
##   rules  FALSE   30      0.7272363  0.5907941
##   rules  FALSE   40      0.7134851  0.5699375
##   rules   TRUE   10      0.6765007  0.5144960
##   rules   TRUE   20      0.6629039  0.4941497
##   rules   TRUE   30      0.6656188  0.4982508
##   rules   TRUE   40      0.6628328  0.4940570
##   tree   FALSE   10      0.7114469  0.5666792
##   tree   FALSE   20      0.6899086  0.5342603
##   tree   FALSE   30      0.6923735  0.5380559
##   tree   FALSE   40      0.6844198  0.5262251
##   tree    TRUE   10      0.6573562  0.4857620
##   tree    TRUE   20      0.6711786  0.5066193
##   tree    TRUE   30      0.6849177  0.5273226
##   tree    TRUE   40      0.6714042  0.5069795
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were trials = 30, model = rules
##  and winnow = FALSE.
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
## 0.9622642 0.9433650
```

```r
confusionMatrix(hat$hat, hat$type)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction Ghost Ghoul Goblin
##     Ghost    117     0      6
##     Ghoul      0   128      7
##     Goblin     0     1    112
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9623          
##                  95% CI : (0.9375, 0.9792)
##     No Information Rate : 0.3477          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9434          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: Ghost Class: Ghoul Class: Goblin
## Sensitivity                1.0000       0.9922        0.8960
## Specificity                0.9764       0.9711        0.9959
## Pos Pred Value             0.9512       0.9481        0.9912
## Neg Pred Value             1.0000       0.9958        0.9496
## Prevalence                 0.3154       0.3477        0.3369
## Detection Rate             0.3154       0.3450        0.3019
## Detection Prevalence       0.3315       0.3639        0.3046
## Balanced Accuracy          0.9882       0.9817        0.9460
```

```r
varImp(trainingModel)
```

```
## C5.0 variable importance
## 
##               Overall
## hair_length    100.00
## bone_length    100.00
## has_soul       100.00
## rotting_flesh  100.00
## colorclear      91.00
## colorgreen      87.89
## colorwhite      86.16
## colorblue       84.43
## colorblood       0.00
```

Display the final model.


```r
trainingModel$finalModel
```

```
## 
## Call:
## C5.0.default(x = structure(c(0.354512184582154,
##  "winnow", "noGlobalPruning", "CF", "minCases",
##  "fuzzyThreshold", "sample", "earlyStopping", "label", "seed")))
## 
## Rule-Based Model
## Number of samples: 371 
## Number of predictors: 9 
## 
## Number of boosting iterations: 30 
## Average number of rules: 14.7 
## 
## Non-standard options: attempt to group attributes
```

---

# Predict on `test`

Apply the model to the `test` data.


```r
hat <-
  test %>%
  transform(type = predict(trainingModel, test)) %>%
  select(matches("id|type"))
dim(hat)
```

```
## [1] 529   2
```

```r
str(hat)
```

```
## 'data.frame':	529 obs. of  2 variables:
##  $ id  : int  3 6 9 10 13 14 15 16 17 18 ...
##  $ type: Factor w/ 3 levels "Ghost","Ghoul",..: 2 2 2 1 1 1 2 3 3 3 ...
```

```r
head(hat)
```

```
##   id  type
## 1  3 Ghoul
## 2  6 Ghoul
## 3  9 Ghoul
## 4 10 Ghost
## 5 13 Ghost
## 6 14 Ghost
```

Describe the `type` variable.


```r
tab <- table(hat$type)
data.frame(cbind(freq = tab, prop = prop.table(tab)))
```

```
##        freq      prop
## Ghost   200 0.3780718
## Ghoul   205 0.3875236
## Goblin  124 0.2344045
```

Save the predictions to file.


```r
options(scipen = 10)
write.csv(hat, file = "../data/processed/submission.csv", row.names = FALSE, quote = FALSE)
file.info("../data/processed/submission.csv")
```

```
##                                  size isdir mode               mtime
## ../data/processed/submission.csv 5899 FALSE  666 2016-11-06 15:06:32
##                                                ctime               atime
## ../data/processed/submission.csv 2016-11-05 10:02:39 2016-11-05 10:02:39
##                                  exe
## ../data/processed/submission.csv  no
```
