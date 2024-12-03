######################################## Replication of Burke 2015 #############################################


## cleanup 
cat("\014") 
graphics.off()  # clear all graphs
rm(list = ls()) # remove all files from your workspace
pacman::p_load(pacman, ggplot2, dplyr, lmtest, margins, parallel, maptools, fields, ClassInt, plotrix, sandwich, webuse, tidyr, readxl, RColorBrewer, reshape2) #installing neccesary packages 
options(scipen = 999, digits = 5)
source("EE/helper.R")


################################### Part 1 #######################################################
#                                                                                                #                
#                   ESTIMATE GLOBAL RESPONSE WITH BASELINE REGRESSION SPECIFICATION              #
#                                                                                                #
##################################################################################################

# Load the updated dataset
dataset <- read_excel("MainData.xlsx")

temp_values <- seq(0, 30, by = 1)
#####################  Caluclating predictive margins for 3 levels of readiness ############################

#loading readiness data in readinessxlsx as a dataframe


readiness <- as.data.frame(read_excel("EE/readiness.xlsx"))

readiness <- melt(readiness, id.vars = c("ISO3", "Name"), variable.name = "Year", value.name = "Readiness")

uniques <-  unique(dataset$CountryCode)

#summarise precipitation, temperature and readiness data  
summary(summ_statistics_precipitation)


readiness$Readiness <- as.numeric(readiness$Readiness)/max(as.numeric(readiness$Readiness), na.rm = TRUE)

readiness_sorted <- sort(readiness$Readiness)


# Calculate deciles for readiness scores
deciles <- quantile(readiness_sorted, probs = seq(0, 1, by = 0.33), na.rm = TRUE)
# Step 4: Plot a histogram of readiness scores across all years
hist(readiness$Readiness, 
     xlab="Readiness Scores", col="skyblue", border="white", breaks=100, xlim=c(0,1), main=NULL)


# add abline for 2 and 3 decile 
abline(v = c(deciles[2], deciles[3]), col = c("black", "black"), lty = 2)


#assuming that the first readiness score is the same as the period before down until 1960. Specifically I want to extend the readiness with the scores from 1960 to 1994.

# Convert Year column to numeric (if it's not already)
readiness$Year <- as.numeric(as.character(readiness$Year))

# Get the first readiness score for each country
first_readiness <- readiness %>%
  group_by(ISO3) %>%
  filter(Year == 1995) %>%
  select(ISO3, Readiness)

# Duplicate the readiness scores for each year from 1960 to 1994
first_readiness <- first_readiness[rep(row.names(first_readiness), each = 35),]

# Adding the years from 1960 to 1994
first_readiness$Year <- rep(1960:1994, nrow(first_readiness) / 35)

# Changing so that the first readiness has the following column structure: ISO3, Year, Readiness
first_readiness <- first_readiness[, c(1, 3, 2)]

# Adding the duplicated readiness scores to the original dataset
readiness <- rbind(readiness[,-c(2)], first_readiness)



#Mergin the datasets, first based on year and then by country 
dataset <- merge(dataset, readiness, by.x = c("CountryCode", "Year"), by.y = c("ISO3", "Year"))

#make new column called ready_low if readiness is in the lowest decile, ready_med if in the middle decile, ready_high if in the highest decile
dataset$ready_low <- ifelse(dataset$Readiness < deciles[2], 1, 0)
dataset$ready_med <- ifelse(dataset$Readiness >= deciles[2] & dataset$Readiness < deciles[3], 1, 0)
dataset$ready_high <- ifelse(dataset$Readiness >= deciles[3], 1, 0)
#define temperature variable
dataset$temp <- dataset$TempPopWeight

#temp squared
dataset$temp_sq <- dataset$temp^2

dataset$precip <- dataset$PrecipPopWeight
dataset$precip_sq <- dataset$precip^2



# Define the formula for the baseline regression

yi_vars <- grep("_yi_", names(dataset), value = TRUE) ## all variables relating to time_trend
y2_vars <- grep("_y2_", names(dataset), value = TRUE) ## all variables relating to squared time_trend

# Combine the variables to include in the regression
all_additional_vars <- c(yi_vars, y2_vars)

# Define the full formula for the baseline regression
full_formula <- update(GrowthWDI ~ temp + temp_sq + precip + precip_sq + 
                         factor(Year) + factor(ISO), 
                       paste(". ~ . +", paste(all_additional_vars, collapse = " + ")))

# Fit the model
  
model_baseline <- lm(
  full_formula,
  data = dataset
)

# define the full formula for the extended regression adding interactions with readiness
full_formula_ready <- update(full_formula, 
                             paste(". ~ . + ready_med*(temp + temp_sq+precip+precip_sq) + ready_high*(temp + temp_sq+precip+precip_sq)"))

model_readiness <- lm(
  full_formula_ready,
  data = dataset
)


tail(coef(summary(model_readiness)), n=10)



#Calculate predictive margins for each readiness level 

# Initialize an empty list to store predictions for each readiness category
all_predictions <- list()

# Loop through each readiness category
#create list with all categories

readiness_categories <- list(
  ready_low = "Low Readiness",
  ready_med = "Medium Readiness",
  ready_high = "High Readiness"
)

new_data<-model_readiness$model
colnames(new_data)[6] <- "Year"
colnames(new_data)[7] <- "ISO"

#initialise loop 

for (readynes_category in names(readiness_categories)) {
  
  # Vectors to store average predictions and confidence intervals
  avg_pred_ready <- numeric(length(temp_values))
  lower_bound <- numeric(length(temp_values))  
  upper_bound <- numeric(length(temp_values))  
  
  # Set readiness category dummies based on the current category
  if (readynes_category == "ready_low") {
    new_data$ready_med <- 0
    new_data$ready_high <- 0
  } else if (readynes_category == "ready_med") {
    new_data$ready_high <- 0
    
  } else if (readynes_category == "ready_high") {
    new_data$ready_med <- 0
  }
  
  
  # Loop through each temperature value to calculate margins
  for (i in seq_along(temp_values)) {
    temp_val <- temp_values[i]
    
    # Fix temp at the current value and calculate squared term
    new_data$temp <- temp_val
    new_data$temp_sq <- temp_val^2
    
    # Make predictions for the current temp and readiness category
    pred <- predict(model_readiness, newdata = new_data, se.fit = TRUE)
    
    # Store the average prediction
    avg_pred_ready[i] <- mean(pred$fit, na.rm = TRUE)
    
  }
  
  # Combine predictions into a data frame for the current readiness category
  results_ready <- data.frame(
    temp = temp_values,
    avg_prediction = avg_pred_ready,
    lower_bound = lower_bound,
    upper_bound = upper_bound,
    readiness_category = rep(readiness_categories[[readynes_category]], length(temp_values))
  )
  
  # Append the results to the list of all predictions
  all_predictions[[length(all_predictions) + 1]] <- results_ready
}

#create final results
final_result <- do.call(rbind, all_predictions)


#plotting each scenario
ggplot(final_result, aes(x = temp, y = avg_prediction, color = readiness_category)) +
  geom_line(size = 0.5) +  # Thicker lines for better visibility
  #geom_ribbon(aes(ymin = lower_bound, ymax = upper_bound, fill = readiness_category), alpha = 0.2) +  # Shaded confidence interval
  labs(title = "Predicted Growth by Temperature for Different Readiness Categories",
       x = "Temperature (°C)",
       y = "Average Predicted Growth",
       color = "Readiness Category", fill = "Readiness Category") +  # Legend labels
  xlim(0, 35) +  # Set x-axis limits
  scale_color_brewer(palette = "Set1") +  # Use distinct colors for lines
  theme_minimal() +
  theme(
    legend.position = "top",  # Move legend to the top for better visibility
    plot.title = element_text(hjust = 0.5)  # Center-align the title
  
  )







######################################## Projections ######################

dir.create("EE/projectionOutput")

years <- 2024:2099
#Baseline


# get the average temperature for each country in dataset
mt <- dataset %>%   #the following few lines gets the average temperature in each country for the years we want, using dplyr
  filter(Year>=1980 & is.na(TempPopWeight)==F & is.na(GrowthWDI)==F) %>% 
  group_by(CountryCode) %>% 
  summarize(meantemp = mean(TempPopWeight,na.rm=T), basegrowth = mean(GrowthWDI, na.rm=T), gdpCap = mean(GDPCap,na.rm=T))





pop <- read.csv("EE/SSP_PopulationProjections.csv")
levels(pop$Scenario)[levels(pop$Scenario)=="SSP4d_v9_130115"] <- "SSP4_v9_130115" 
growth <- read.csv("EE/SSP_GrowthProjections.csv")
temp <- read.csv("EE/CountryTempChange_RCP85.csv")

# Step 1: Find the common CountryCodes in all three datasets
common_countries <- Reduce(intersect, list(temp$GMI_CNTRY, growth$Region, pop$Region))


#Growth scenario
growth1 <- ipolate(growth)
growth1[,names(growth1)%in%years] = growth1[,names(growth1)%in%years]/100 # scaling all values dówn with 100
growthSSP <- growth1 %>% filter(Scenario == "SSP1_v9_130325")
growthSSP <- merge(mt, growthSSP[3:ncol(growthSSP)], by.x="CountryCode", by.y="Region")
growthSSP <- growthSSP[growthSSP$CountryCode %in% common_countries,]

# we have 162 matching countries from growth


#population scenario 
pop1 <- ipolate(pop)
popSSP <- pop1 %>% filter(Scenario == "SSP1_v9_130115")
popSSP <- merge(mt,popSSP[3:ncol(popSSP)],by.x="CountryCode",by.y="Region")
popSSP <- popSSP[popSSP$CountryCode %in% common_countries,]

#temp scenario
temp1 <- merge(popSSP[,1:3], temp, by.x="CountryCode", by.y="GMI_CNTRY")
temp1 <- temp1[temp1$CNTRY_NAME%in%c("West Bank","Gaza Strip","Bouvet Island")==F,]
temp1 <- temp1[temp1$CountryCode %in% common_countries, ]
deltaTemp <- temp1$Tchg

ccd <- deltaTemp/length(years)  #rate of increase in temperature per year.  


basegdp <- growthSSP$gdpCap  #baseline GDP/cap
temp <- growthSSP$meantemp  #baseline temperature.  

GDPcapCC = GDPcapNoCC = array(dim=c(dim(popSSP)[1],length(years)))  #array to fill with GDP/cap for each country
dimnames(GDPcapCC) <-dimnames(GDPcapNoCC) <- list(popSSP[,1],years)
GDPcapCC[,1] <-  GDPcapNoCC[,1] <- basegdp  #initialize with baseline per cap GDP

temp_coef = 0.032 #coefficient for temperature in growth regression.
temp_sq_coef = -0.001 #coefficient for temperature squared in growth regression.
  bg = temp_coef*temp + temp_sq_coef*temp*temp  #this finds the predicted growth level for each country's temperature

  for (i in 2:length(years)) {
    j = i - 1
    y = years[i]
    basegrowth <- growthSSP[,which(names(growthSSP)==y)]  #growth rate without climate change
    GDPcapNoCC[,i] = GDPcapNoCC[,j]*(1+basegrowth)  #last year's per cap GDP times this years growth rate, as projected by scenario
    newtemp = temp+j*ccd
    dg = temp_coef*newtemp + temp_sq_coef*newtemp*newtemp  #predicted growth under new temperature
    dg[newtemp>30] = temp_coef*30 + temp_sq_coef*30*30  #constrain response to response at 30C if temp goes above that.  this is so we are not projecting out of sample
    
    diff = dg - bg  #difference between predicted baseline growth and predicted growth under new temp
    GDPcapCC[,i] = GDPcapCC[,j]*(1+basegrowth + diff)  #last year's GDPcap (w/ climate change) times climate-adjusted growth rate for this year
  }
  
GDPcapCC <- melt(GDPcapCC)
names(GDPcapCC) <- c("CountryCode", "Year", "Value")

GDPcapNoCC <- melt(GDPcapNoCC)
names(GDPcapNoCC) <- c("CountryCode", "Year", "Value")

latest_status <- dataset %>%
  group_by(CountryCode) %>%
  filter(Year == max(Year)) %>%   # Get the latest year data
  mutate(
    Latest_DemocracyStatus = case_when(
      Readiness < deciles[2] ~ "Low",
      Readiness >= deciles[2] & Readiness < deciles[3] ~ "Medium",
      TRUE ~ "High"  # Anything greater than or equal to deciles[3]
    )
  ) %>%
  select(CountryCode, Latest_DemocracyStatus)
GDPcapNoCC <- merge(GDPcapNoCC, latest_status, by = "CountryCode", all.x = TRUE)
GDPcapCC <- merge(GDPcapCC, latest_status, by = "CountryCode", all.x = TRUE)

GDPcapCC <- GDPcapCC[order(GDPcapCC$CountryCode, GDPcapCC$Year),]

GDPcapNoCC <-GDPcapNoCC[order(GDPcapCC$CountryCode, GDPcapCC$Year),]

GDPcapCC <- cbind(GDPcapCC, pct_change=0)



for (year in years) {
  GDPcapCC[which(GDPcapCC$Year==year),]["pct_change"] <- 100 * (GDPcapCC[which(GDPcapCC$Year==year),]["Value"] - GDPcapNoCC[which(GDPcapNoCC$Year==year),]["Value"] ) / GDPcapCC[which(GDPcapCC$Year==year),]["Value"] 
}

tail(GDPcapCC)


averages <- GDPcapCC %>% group_by(Year, Latest_DemocracyStatus) %>%
 summarise(mean=mean(pct_change))


ggplot(averages, aes(x = Year, y = mean, color = Latest_DemocracyStatus)) +
  geom_line(size = 1.2) +
  labs(title = "Average Percentage Change in GDP Relative to constant temperatures (RCP 8.5, SSP1)",
       x = "Year",
       y = "Average % Change in GDP",
       color = "Readiness Status") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")




#making robustness checks 

#excluding oil countries: https://www.theglobaleconomy.com/rankings/oil_revenue/ (countries with oil revenue > 10% of GDP)
oil_countries <- c("ARE", "BHR", "DZA", "EGY", "IRN", "IRQ", "KWT", "LBY", "OMN", "QAT", "SAU", "SDN", "SYR", "TUN", "YEM")

# excluding oil countries from data: 
dataset_no_oil <- dataset[!dataset$CountryCode %in% oil_countries,]

model_oil <- lm(
  full_formula_ready,
  data = dataset_no_oil
)

tail(coef(summary(model_oil)))


OECD_countries <- c("AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CZE", "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "ISL", "IRL", "ISR", "ITA", "JPN", "KOR", "LVA", "LTU", "LUX", "MEX", "NLD", "NZL", "NOR", "POL", "PRT", "SVK", "SVN", "ESP", "SWE", "CHE", "TUR", "GBR", "USA")

dataset_oecd <- dataset[dataset$CountryCode %in% OECD_countries,]

model_oecd <- lm(
  full_formula_ready,
  data = dataset_oecd
)

tail(coef(summary(model_oecd)))
    