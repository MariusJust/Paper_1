## cleanup 
cat("\014") 
graphics.off()  # clear all graphs
rm(list = ls()) # remove all files from your workspace
pacman::p_load(pacman, ggplot2, dplyr, lmtest, margins, parallel, maptools, fields, ClassInt, plotrix, sandwich, webuse, tidyr, readxl) #installing neccesary packages 
options(scipen = 999, digits = 5)


rawData <- read_xlsx("EE- P subject/Raw.xlsx", sheet=2)

countries <- rawData$`Survey Edition`[3:nrow(rawData)]
yearData <- rep(rep(1973:2023), times= length(countries))

country_rep <- rep(countries, each = length(rep(1973:2023)))

democracy_columns <- seq(4, ncol(rawData), by = 3)
democracy_scores <- rawData[3:nrow(rawData), democracy_columns]
flatten_democracy <- as.vector(t(democracy_scores))

dset <- data.frame(
  Year=yearData,
  Country = country_rep,
  Dem_Status = flatten_democracy
)

write.csv(dset, "EE- P subject/democracy.csv")