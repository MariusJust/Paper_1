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
dataset <- read_excel("Repo/EE/MainData.xlsx")

temp_values <- seq(-5, 35, by = 1)
#####################  Caluclating predictive margins for 3 democracy levels ############################
dem_dataset <- dataset[dataset$DemocracyStatus != "NA" & dataset$DemocracyStatus != "-",]
dem_dataset$precip<-dem_dataset$PrecipPopWeight
dem_dataset$precip_sq<-dem_dataset$precip^2

# Define the temperature variable
dem_dataset$temp <- dem_dataset$TempPopWeight
# Define the quadratic term for temperature
dem_dataset$temp_sq <- dem_dataset$temp^2


yi_vars <- grep("_yi_", names(dem_dataset), value = TRUE) ## all variables relating to time_trend
y2_vars <- grep("_y2_", names(dem_dataset), value = TRUE) ## all variables relating to squared time_trend

# Combine the variables to include in the regression
all_additional_vars <- c(yi_vars, y2_vars)



# summary_stats_precip <- dem_dataset %>%
#   group_by(Democracy_Status) %>%
#   summarise(
#     mean_precip = mean(precip, na.rm = TRUE),
#     median_precip = median(precip, na.rm = TRUE),
#     sd_precip = sd(precip, na.rm = TRUE),
#     min_precip = min(precip, na.rm = TRUE),
#     max_precip = max(precip, na.rm = TRUE),
#     n = n()
#   )
# 



### average precipitation for the democracy categories 
# Modify the formula to include interactions with democracy status
full_formula <- update(GrowthWDI ~ temp + temp_sq + precip + precip_sq + 
                         Dem_NotFree * (temp + temp_sq + precip + precip_sq) +
                         Dem_PartiallyFree * (temp + temp_sq + precip + precip_sq) +
                         factor(Year) + factor(ISO), 
                       paste(". ~ . +", paste(all_additional_vars, collapse = " + ")))

# Fit the model
model_democracy <- lm(
  full_formula,
  data = dem_dataset
)


temp_coef <- coef(model_democracy)[2]
temp_sq_coef <- coef(model_democracy)[3]

# Step 2: Use the data that was used in the model as the "new_data" in the prediction function
new_data <- model_democracy$model  # Use the model's data directly
colnames(new_data)[8] <- "Year"
colnames(new_data)[9] <- "ISO"

# Define democracy categories with labels
democracy_categories <- list(
  dem_free = "F",  # Free category (reference)
  dem_notFree = "NF",  # Not Free category
  dem_partiallyFree = "PF"  # Partially Free category
)

# # Initialize an empty list to store predictions for each democracy category
# all_predictions <- list()
# 
# # Loop through each democracy category
# for (dem_category in names(democracy_categories)) {
#   
#   # Vectors to store average predictions and confidence intervals
#   avg_pred_dem <- numeric(length(temp_values))
#   lower_bound <- numeric(length(temp_values))  
#   upper_bound <- numeric(length(temp_values))  
#   
#   # Loop through each temperature value to calculate margins
#   for (i in seq_along(temp_values)) {
#     temp_val <- temp_values[i]
#     
#     # Fix temp at the current value and calculate squared term
#     new_data$temp <- temp_val
#     new_data$temp_sq <- temp_val^2
#     
#     # Set democracy category dummies based on the current category
#     if (dem_category == "dem_free") {
#       new_data$Dem_NotFree <- 0
#       new_data$Dem_PartiallyFree <- 0
#     } else if (dem_category == "dem_notFree") {
#       new_data$Dem_NotFree <- 1
#       new_data$Dem_PartiallyFree <- 0
#     } else if (dem_category == "dem_partiallyFree") {
#       new_data$Dem_NotFree <- 0
#       new_data$Dem_PartiallyFree <- 1
#     }
#     
#     # Make predictions for the current temp and democracy category
#     pred <- predict(model_democracy, newdata = new_data, se.fit = TRUE)
#     
#     # Store the average prediction
#     avg_pred_dem[i] <- mean(pred$fit, na.rm = TRUE)
#     
#     # 95% confidence interval (z-value for 95% CI is 1.96)
#     ci_multiplier <- 1.64
#     lower_bound[i] <- mean(pred$fit - ci_multiplier * pred$se.fit, na.rm = TRUE)
#     upper_bound[i] <- mean(pred$fit + ci_multiplier * pred$se.fit, na.rm = TRUE)
#   }
#   
#   # Combine predictions into a data frame for the current democracy category
#   results_dem <- data.frame(
#     temp = temp_values,
#     avg_prediction = avg_pred_dem,
#     lower_bound = lower_bound,
#     upper_bound = upper_bound,
#     democracy_category = rep(democracy_categories[[dem_category]], length(temp_values))
#   )
#   
#   # Append the results to the list of all predictions
#   all_predictions[[length(all_predictions) + 1]] <- results_dem
# }
# 
# # Combine all predictions into one data frame
# final_results <- do.call(rbind, all_predictions)
# 
# 
# 
# # Plottting all 3 democracy scenarios, without confidence intervals
# ggplot(final_results, aes(x = temp, y = avg_prediction, color = democracy_category)) +
#   geom_line(size = 0.5) +  # Thicker lines for better visibility
#   #geom_ribbon(aes(ymin = lower_bound, ymax = upper_bound, fill = democracy_category), alpha = 0.2) +  # Shaded confidence interval
#   labs(title = "Predicted Growth by Temperature for Different Democracy Categories",
#        x = "Temperature (°C)",
#        y = "Average Predicted Growth",
#        color = "Democracy Category", fill = "Democracy Category") +  # Legend labels
#   xlim(0, 35) +  # Set x-axis limits
#   scale_color_brewer(palette = "Set1") +  # Use distinct colors for lines
#   scale_fill_brewer(palette = "Set1") +  # Use matching fill colors for confidence intervals
#   theme_minimal() +
#   theme(
#     legend.position = "top",  # Move legend to the top for better visibility
#     plot.title = element_text(hjust = 0.5)  # Center-align the title
#   )
#     
# 





######################################## Projections ######################

dir.create("EE/projectionOutput")

years <- 2024:2099
#Baseline


latest_status <- dem_dataset %>% 
  filter(Year == 2023) %>%          # Select only the data from the latest year (2023)
  select(CountryCode, DemocracyStatus) %>%  # Select relevant columns
  rename(Latest_DemocracyStatus = DemocracyStatus) # Rename for clarity


dem_dataset <- merge(dem_dataset, latest_status, by = "CountryCode")

# Filter for free countries
free_countries <- dem_dataset %>% filter(Latest_DemocracyStatus == "F")

# Filter for not free countries
not_free_countries <- dem_dataset %>% filter(Latest_DemocracyStatus == "NF")

partially_free_countries <- dem_dataset %>% filter(Latest_DemocracyStatus == "PF")

mt <- dem_dataset %>%   #the following few lines gets the average temperature in each country for the years we want, using dplyr
  filter(Year>=1980 & is.na(TempPopWeight)==F & is.na(GrowthWDI)==F) %>% 
  group_by(CountryCode) %>% 
  summarize(meantemp = mean(TempPopWeight,na.rm=T), basegrowth = mean(GrowthWDI, na.rm=T), gdpCap = mean(GDPCap,na.rm=T))
mt <- as.data.frame(mt)



## loading scenario data

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



# Merge democracy status for free countries

GDPcapNoCC <- merge(GDPcapNoCC, latest_status, by = "CountryCode", all.x = TRUE)
GDPcapCC <- merge(GDPcapCC, latest_status, by = "CountryCode", all.x = TRUE)

GDPcapCC <- GDPcapCC[order(GDPcapCC$CountryCode, GDPcapCC$Year),]

GDPcapNoCC <-GDPcapNoCC[order(GDPcapCC$CountryCode, GDPcapCC$Year),]

GDPcapCC <- cbind(GDPcapCC, pct_change=0)

for (year in years) {
  GDPcapCC[which(GDPcapCC$Year==year),]["pct_change"] <- 100 * (GDPcapCC[which(GDPcapCC$Year==year),]["Value"] - GDPcapNoCC[which(GDPcapNoCC$Year==year),]["Value"] ) / GDPcapCC[which(GDPcapCC$Year==year),]["Value"] 
}

tail(GDPcapCC)


averages <- GDPcapCC %>% group_by( Year, Latest_DemocracyStatus) %>%
 summarise(mean=mean(pct_change))


ggplot(averages[which(averages$Latest_DemocracyStatus %in%  c("F","NF")),], aes(x = Year, y = mean, color = Latest_DemocracyStatus)) +
  geom_line(size = 1.2) +
  labs(title = "Average Percentage Change in GDP Relative to constant temperatures (RCP 8.5, SSP1)",
       x = "Year",
       y = "Average % Change in GDP",
       color = "Democracy Status") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")


    