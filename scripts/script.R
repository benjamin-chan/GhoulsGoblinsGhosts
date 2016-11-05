library(checkpoint)
checkpoint("2016-10-01", use.knitr = TRUE)

setwd("~/Projects/Kaggle/GhoulsGoblinsGhosts/scripts")

Sys.time0 <- Sys.time()

sink("script.log")
files <- c("header.yaml",
           "preamble.Rmd",
           "read.Rmd",
           "explore.Rmd",
           "model.Rmd",
           "predict.Rmd")
f <- file("master.Rmd", open = "w")
for (i in 1:length(files)) {
    x <- readLines(files[i])
    writeLines(x, f)
    if (i < length(files)) {writeLines("\n---\n", f)}
}
close(f)
library(knitr)
library(rmarkdown)
opts_chunk$set(fig.path = "../figures/")
knit("master.Rmd", output = "../docs/index.md")
# pandoc("../docs/index.md", format = "html")
file.remove("master.Rmd")
sink()

sink("session.log")
list(completionDateTime = Sys.time(),
     executionTime = Sys.time() - Sys.time0,
     sessionInfo = sessionInfo())
sink()
