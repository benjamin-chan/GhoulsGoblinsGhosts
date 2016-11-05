---
title: "Kaggle: Ghouls, Goblins, and Ghosts... Boo!"
date: "2016-11-05 10:38:39"
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
library(xgboost)
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
##  [1] rmarkdown_1.0       knitr_1.14          checkpoint_0.3.16  
##  [4] randomForest_4.6-12 doParallel_1.0.10   iterators_1.0.8    
##  [7] foreach_1.4.3       xgboost_0.4-3       caret_6.0-71       
## [10] lattice_0.20-33     ggplot2_2.1.0       reshape2_1.4.1     
## [13] dplyr_0.5.0         plyr_1.8.4          RevoUtilsMath_8.0.3
## 
## loaded via a namespace (and not attached):
##  [1] Rcpp_0.12.6        formatR_1.4        compiler_3.3.1    
##  [4] nloptr_1.0.4       class_7.3-14       tools_3.3.1       
##  [7] digest_0.6.10      lme4_1.1-12        evaluate_0.9      
## [10] tibble_1.0         nlme_3.1-128       gtable_0.2.0      
## [13] mgcv_1.8-12        Matrix_1.2-6       DBI_0.5           
## [16] SparseM_1.7        e1071_1.6-7        stringr_1.0.0     
## [19] RevoUtils_10.0.1   MatrixModels_0.4-1 stats4_3.3.1      
## [22] grid_3.3.1         nnet_7.3-12        data.table_1.9.6  
## [25] R6_2.1.2           minqa_1.2.4        car_2.1-3         
## [28] magrittr_1.5       htmltools_0.3.5    scales_0.4.0      
## [31] codetools_0.2-14   MASS_7.3-45        splines_3.3.1     
## [34] assertthat_0.1     pbkrtest_0.4-6     colorspace_1.2-6  
## [37] labeling_0.3       quantreg_5.26      stringi_1.1.1     
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
                     allowParallel = FALSE)
```

Set the model.


```r
method <- "xgbTree"
```

Set the tuning grid for model xgbTree.


```r
grid <- expand.grid(nrounds = seq(50, 150, 50), 
                    max_depth = 1:3, 
                    eta = seq(0.1, 0.3, 0.1), 
                    gamma = 0, 
                    colsample_bytree = seq(0.2, 0.4, 0.1), 
                    min_child_weight = 1)
```

Fit model over the tuning parameters.


```r
trainingModel <- train(type ~ .,
                       data = select(train, -matches("id")),
                       method = method,
                       trControl = ctrl,
                       tuneGrid = grid)
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
##   eta  max_depth  colsample_bytree  nrounds  Accuracy   Kappa    
##   0.1  1          0.2                50      0.6846405  0.5264453
##   0.1  1          0.2               100      0.7041407  0.5559184
##   0.1  1          0.2               150      0.7171476  0.5753904
##   0.1  1          0.3                50      0.7092837  0.5638111
##   0.1  1          0.3               100      0.7215008  0.5819990
##   0.1  1          0.3               150      0.7263088  0.5890873
##   0.1  1          0.4                50      0.7136246  0.5702231
##   0.1  1          0.4               100      0.7217221  0.5822780
##   0.1  1          0.4               150      0.7231248  0.5843758
##   0.1  2          0.2                50      0.7104881  0.5650061
##   0.1  2          0.2               100      0.7246999  0.5864925
##   0.1  2          0.2               150      0.7222025  0.5827729
##   0.1  2          0.3                50      0.7230604  0.5840849
##   0.1  2          0.3               100      0.7281651  0.5918070
##   0.1  2          0.3               150      0.7195571  0.5789039
##   0.1  2          0.4                50      0.7260243  0.5885685
##   0.1  2          0.4               100      0.7224937  0.5832881
##   0.1  2          0.4               150      0.7157639  0.5731792
##   0.1  3          0.2                50      0.7023159  0.5530317
##   0.1  3          0.2               100      0.7055860  0.5579283
##   0.1  3          0.2               150      0.6999147  0.5493412
##   0.1  3          0.3                50      0.7177092  0.5762148
##   0.1  3          0.3               100      0.7115403  0.5668790
##   0.1  3          0.3               150      0.7090762  0.5631094
##   0.1  3          0.4                50      0.7249808  0.5872579
##   0.1  3          0.4               100      0.7130307  0.5691668
##   0.1  3          0.4               150      0.7071567  0.5602647
##   0.2  1          0.2                50      0.7053273  0.5576060
##   0.2  1          0.2               100      0.7190712  0.5781851
##   0.2  1          0.2               150      0.7260251  0.5885516
##   0.2  1          0.3                50      0.7191273  0.5784392
##   0.2  1          0.3               100      0.7249879  0.5871099
##   0.2  1          0.3               150      0.7243923  0.5862504
##   0.2  1          0.4                50      0.7198895  0.5795603
##   0.2  1          0.4               100      0.7195801  0.5790194
##   0.2  1          0.4               150      0.7225768  0.5835150
##   0.2  2          0.2                50      0.7193433  0.5787017
##   0.2  2          0.2               100      0.7186569  0.5774886
##   0.2  2          0.2               150      0.7122546  0.5678069
##   0.2  2          0.3                50      0.7251458  0.5873803
##   0.2  2          0.3               100      0.7106631  0.5654930
##   0.2  2          0.3               150      0.6995990  0.5488651
##   0.2  2          0.4                50      0.7241442  0.5857723
##   0.2  2          0.4               100      0.7090633  0.5630517
##   0.2  2          0.4               150      0.6988349  0.5476892
##   0.2  3          0.2                50      0.7015695  0.5518752
##   0.2  3          0.2               100      0.6983828  0.5468856
##   0.2  3          0.2               150      0.6888969  0.5326061
##   0.2  3          0.3                50      0.7123748  0.5681706
##   0.2  3          0.3               100      0.7004470  0.5501483
##   0.2  3          0.3               150      0.6913577  0.5364781
##   0.2  3          0.4                50      0.7147635  0.5717997
##   0.2  3          0.4               100      0.7023839  0.5530787
##   0.2  3          0.4               150      0.6939498  0.5404175
##   0.3  1          0.2                50      0.7112466  0.5665877
##   0.3  1          0.2               100      0.7250693  0.5872039
##   0.3  1          0.2               150      0.7227810  0.5837480
##   0.3  1          0.3                50      0.7303514  0.5952699
##   0.3  1          0.3               100      0.7252573  0.5875707
##   0.3  1          0.3               150      0.7255620  0.5879912
##   0.3  1          0.4                50      0.7238839  0.5854823
##   0.3  1          0.4               100      0.7211950  0.5813434
##   0.3  1          0.4               150      0.7171393  0.5753313
##   0.3  2          0.2                50      0.7181861  0.5768382
##   0.3  2          0.2               100      0.7090051  0.5629267
##   0.3  2          0.2               150      0.7007335  0.5505004
##   0.3  2          0.3                50      0.7243430  0.5860511
##   0.3  2          0.3               100      0.7005288  0.5502646
##   0.3  2          0.3               150      0.6894555  0.5335677
##   0.3  2          0.4                50      0.7179940  0.5764965
##   0.3  2          0.4               100      0.7012828  0.5513712
##   0.3  2          0.4               150      0.6905663  0.5352514
##   0.3  3          0.2                50      0.6958409  0.5431816
##   0.3  3          0.2               100      0.6892208  0.5332118
##   0.3  3          0.2               150      0.6817039  0.5218734
##   0.3  3          0.3                50      0.7039964  0.5554216
##   0.3  3          0.3               100      0.6862107  0.5287089
##   0.3  3          0.3               150      0.6788905  0.5176384
##   0.3  3          0.4                50      0.7086318  0.5625464
##   0.3  3          0.4               100      0.6969674  0.5449793
##   0.3  3          0.4               150      0.6921820  0.5377702
## 
## Tuning parameter 'gamma' was held constant at a value of 0
## 
## Tuning parameter 'min_child_weight' was held constant at a value of 1
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were nrounds = 50, max_depth = 1,
##  eta = 0.3, gamma = 0, colsample_bytree = 0.3 and min_child_weight = 1.
```

```r
ggplot(trainingModel)
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
## 0.8032345 0.7046372
```

```r
confusionMatrix(hat$hat, hat$type)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction Ghost Ghoul Goblin
##     Ghost    105     2     11
##     Ghoul      0   104     25
##     Goblin    12    23     89
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8032          
##                  95% CI : (0.7591, 0.8425)
##     No Information Rate : 0.3477          
##     P-Value [Acc > NIR] : <2e-16          
##                                           
##                   Kappa : 0.7046          
##  Mcnemar's Test P-Value : 0.5465          
## 
## Statistics by Class:
## 
##                      Class: Ghost Class: Ghoul Class: Goblin
## Sensitivity                0.8974       0.8062        0.7120
## Specificity                0.9488       0.8967        0.8577
## Pos Pred Value             0.8898       0.8062        0.7177
## Neg Pred Value             0.9526       0.8967        0.8543
## Prevalence                 0.3154       0.3477        0.3369
## Detection Rate             0.2830       0.2803        0.2399
## Detection Prevalence       0.3181       0.3477        0.3342
## Balanced Accuracy          0.9231       0.8514        0.7849
```

```r
varImp(trainingModel)
```

```
## xgbTree variable importance
## 
##                 Overall
## has_soul      100.00000
## hair_length    99.99853
## bone_length    36.26458
## rotting_flesh  34.18647
## colorblood      0.91183
## colorwhite      0.39462
## colorclear      0.37879
## colorblue       0.08393
## colorgreen      0.00000
```

Display the final model.


```r
trainingModel$finalModel
```

```
## $handle
## <pointer: 0x0000000012c9d990>
## attr(,"class")
## [1] "xgb.Booster.handle"
## 
## $raw
##    [1] 00 00 00 3f 09 00 00 00 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##   [24] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##   [47] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##   [70] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##   [93] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##  [116] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 0e 00
##  [139] 00 00 00 00 00 00 6d 75 6c 74 69 3a 73 6f 66 74 70 72 6f 62 06 00 00
##  [162] 00 00 00 00 00 67 62 74 72 65 65 96 00 00 00 00 00 00 00 09 00 00 00
##  [185] 00 00 00 00 00 00 00 00 00 00 00 00 03 00 00 00 00 00 00 00 00 00 00
##  [208] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##  [231] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##  [254] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##  [277] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##  [300] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##  [323] 00 00 00 00 00 00 00 00 00 00 01 00 00 00 03 00 00 00 00 00 00 00 00
##  [346] 00 00 00 09 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##  [369] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##  [392] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##  [415] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##  [438] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##  [461] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ff ff ff
##  [484] ff 01 00 00 00 02 00 00 00 07 00 00 80 00 00 28 b7 00 00 00 80 ff ff
##  [507] ff ff ff ff ff ff 00 00 00 00 2b f6 7f bc 00 00 00 00 ff ff ff ff ff
##  [530] ff ff ff 00 00 00 00 db ec 79 3c 4a 8c 3a 3e 8e e3 24 43 bd 9b 24 bd
##  [553] 02 00 00 00 00 00 00 00 e3 38 12 43 23 4d 55 bd 00 00 00 00 00 00 00
##  [576] 00 55 55 95 41 61 45 50 3d 00 00 00 00 01 00 00 00 03 00 00 00 00 00
##  [599] 00 00 00 00 00 00 09 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##  [622] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##  [645] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##  [668] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##  [691] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##  [714] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##  [737] ff ff ff ff 01 00 00 00 02 00 00 00 05 00 00 80 00 00 28 b7 00 00 00
##  [760] 80 ff ff ff ff ff ff ff ff 00 00 00 00 c7 e7 30 3c 00 00 00 00 ff ff
##  [783] ff ff ff ff ff ff 00 00 00 00 55 7a 2d bc 5b 3f 35 3d 8e e3 24 43 b5
##  [806] af 03 3d 02 00 00 00 00 00 00 00 c6 71 1c 43 d0 6b 13 3d 00 00 00 00
##  [829] 00 00 00 00 71 1c 07 41 9c 90 10 bd 00 00 00 00 01 00 00 00 03 00 00
##  [852] 00 00 00 00 00 00 00 00 00 09 00 00 00 00 00 00 00 00 00 00 00 00 00
##  [875] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##  [898] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##  [921] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##  [944] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##  [967] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
##  [990] 00 00 00 ff ff ff ff 01 00 00 00
##  [ reached getOption("max.print") -- omitted 38332 entries ]
## 
## $xNames
## [1] "bone_length"   "rotting_flesh" "hair_length"   "has_soul"     
## [5] "colorblood"    "colorblue"     "colorclear"    "colorgreen"   
## [9] "colorwhite"   
## 
## $problemType
## [1] "Classification"
## 
## $tuneValue
##    nrounds max_depth eta gamma colsample_bytree min_child_weight
## 58      50         1 0.3     0              0.3                1
## 
## $obsLevels
## [1] "Ghost"  "Ghoul"  "Goblin"
## attr(,"ordered")
## [1] FALSE
## 
## attr(,"class")
## [1] "xgb.Booster"
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
##  $ type: Factor w/ 3 levels "Ghost","Ghoul",..: 2 3 2 1 1 1 2 3 3 3 ...
```

```r
head(hat)
```

```
##   id   type
## 1  3  Ghoul
## 2  6 Goblin
## 3  9  Ghoul
## 4 10  Ghost
## 5 13  Ghost
## 6 14  Ghost
```

Describe the `type` variable.


```r
tab <- table(hat$type)
data.frame(cbind(freq = tab, prop = prop.table(tab)))
```

```
##        freq      prop
## Ghost   189 0.3572779
## Ghoul   171 0.3232514
## Goblin  169 0.3194707
```

Save the predictions to file.


```r
options(scipen = 10)
write.csv(hat, file = "../data/processed/submission.csv", row.names = FALSE, quote = FALSE)
file.info("../data/processed/submission.csv")
```

```
##                                  size isdir mode               mtime
## ../data/processed/submission.csv 5944 FALSE  666 2016-11-05 10:52:36
##                                                ctime               atime
## ../data/processed/submission.csv 2016-11-05 08:42:09 2016-11-05 08:42:09
##                                  exe
## ../data/processed/submission.csv  no
```
