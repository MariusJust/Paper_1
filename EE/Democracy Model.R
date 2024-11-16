######################################## Replication of Burke 2015 #############################################


## cleanup 
cat("\014") 
graphics.off()  # clear all graphs
rm(list = ls()) # remove all files from your workspace
pacman::p_load(pacman, ggplot2, dplyr, lmtest, margins, parallel, maptools, fields, ClassInt, plotrix, sandwich, webuse, tidyr, readxl, RColorBrewer) #installing neccesary packages 
options(scipen = 999, digits = 5)
source("EE/helper.R")


################################### Part 1 #######################################################
#                                                                                                #                
#                   ESTIMATE GLOBAL RESPONSE WITH BASELINE REGRESSION SPECIFICATION              #
#                                                                                                #
##################################################################################################

# Load the updated dataset
dataset <- read_excel("MainData.xlsx")

temp_values <- seq(-5, 35, by = 1)
#####################  Caluclating predictive margins for 3 democracy levels ############################
dem_dataset <- dataset[dataset$Democracy_Status != "NA" & dataset$Democracy_Status != "-",]
dem_dataset$precip<-dem_dataset$Precip_PopWeight
dem_dataset$precip_sq<-dem_dataset$precip^2

# Define the temperature variable
dem_dataset$temp <- dem_dataset$Temp_PopWeight
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
full_formula <- update(growthWDI ~ temp + temp_sq + precip + precip_sq + 
                         Dem_NotFree * (temp + temp_sq + precip + precip_sq) +
                         Dem_PartiallyFree * (temp + temp_sq + precip + precip_sq) +
                         factor(Year) + factor(ISO), 
                       paste(". ~ . +", paste(all_additional_vars, collapse = " + ")))

# Fit the model
model_democracy <- lm(
  full_formula,
  data = dem_dataset
)


#Only showing summary statistics for variables of interest
summary(model_democracy)$coefficients[c("temp", "temp_sq", "precip", "precip_sq", "Dem_NotFree", "Dem_PartiallyFree"),]

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

# Initialize an empty list to store predictions for each democracy category
all_predictions <- list()

# Loop through each democracy category
for (dem_category in names(democracy_categories)) {
  
  # Vectors to store average predictions and confidence intervals
  avg_pred_dem <- numeric(length(temp_values))
  lower_bound <- numeric(length(temp_values))  
  upper_bound <- numeric(length(temp_values))  
  
  # Loop through each temperature value to calculate margins
  for (i in seq_along(temp_values)) {
    temp_val <- temp_values[i]
    
    # Fix temp at the current value and calculate squared term
    new_data$temp <- temp_val
    new_data$temp_sq <- temp_val^2
    
    # Set democracy category dummies based on the current category
    if (dem_category == "dem_free") {
      new_data$Dem_NotFree <- 0
      new_data$Dem_PartiallyFree <- 0
    } else if (dem_category == "dem_notFree") {
      new_data$Dem_NotFree <- 1
      new_data$Dem_PartiallyFree <- 0
    } else if (dem_category == "dem_partiallyFree") {
      new_data$Dem_NotFree <- 0
      new_data$Dem_PartiallyFree <- 1
    }
    
    # Make predictions for the current temp and democracy category
    pred <- predict(model_democracy, newdata = new_data, se.fit = TRUE)
    
    # Store the average prediction
    avg_pred_dem[i] <- mean(pred$fit, na.rm = TRUE)
    
    # 95% confidence interval (z-value for 95% CI is 1.96)
    ci_multiplier <- 1.64
    lower_bound[i] <- mean(pred$fit - ci_multiplier * pred$se.fit, na.rm = TRUE)
    upper_bound[i] <- mean(pred$fit + ci_multiplier * pred$se.fit, na.rm = TRUE)
  }
  
  # Combine predictions into a data frame for the current democracy category
  results_dem <- data.frame(
    temp = temp_values,
    avg_prediction = avg_pred_dem,
    lower_bound = lower_bound,
    upper_bound = upper_bound,
    democracy_category = rep(democracy_categories[[dem_category]], length(temp_values))
  )
  
  # Append the results to the list of all predictions
  all_predictions[[length(all_predictions) + 1]] <- results_dem
}

# Combine all predictions into one data frame
final_results <- do.call(rbind, all_predictions)



# Plottting all 3 democracy scenarios, without confidence intervals
ggplot(final_results, aes(x = temp, y = avg_prediction, color = democracy_category)) +
  geom_line(size = 0.5) +  # Thicker lines for better visibility
  #geom_ribbon(aes(ymin = lower_bound, ymax = upper_bound, fill = democracy_category), alpha = 0.2) +  # Shaded confidence interval
  labs(title = "Predicted Growth by Temperature for Different Democracy Categories",
       x = "Temperature (°C)",
       y = "Average Predicted Growth",
       color = "Democracy Category", fill = "Democracy Category") +  # Legend labels
  xlim(0, 35) +  # Set x-axis limits
  scale_color_brewer(palette = "Set1") +  # Use distinct colors for lines
  scale_fill_brewer(palette = "Set1") +  # Use matching fill colors for confidence intervals
  theme_minimal() +
  theme(
    legend.position = "top",  # Move legend to the top for better visibility
    plot.title = element_text(hjust = 0.5)  # Center-align the title
  )
    






######################################## intercacting democracy status with rich/poor ######################






######################################## Projections ######################

dir.create("EE/projectionOutput")
dem_dataset <- dataset[dataset$Democracy_Status != "NA" & dataset$Democracy_Status != "-",]


#Baseline
gdpCap <- dem_dataset$`GDP per capita (constant 2015 US$)`
dem_dataset <- data.frame(dem_dataset,gdpCap)
mt <- dem_dataset %>%   #the following few lines gets the average temperature in each country for the years we want, using dplyr
  filter(Year>=1980 & is.na(Temp_PopWeight)==F & is.na(growthWDI)==F) %>% 
  group_by(Country.Code) %>% 
  summarize(meantemp = mean(Temp_PopWeight,na.rm=T), basegrowth = mean(growthWDI, na.rm=T), gdpCap = mean(`GDP per capita (constant 2015 US$)`,na.rm=T))
mt <- as.data.frame(mt)
yrs <- 2024:2099


pop <- read.csv("EE/SSP_PopulationProjections.csv")
levels(pop$Scenario)[levels(pop$Scenario)=="SSP4d_v9_130115"] <- "SSP4_v9_130115" 
growth <- read.csv("EE/SSP_GrowthProjections.csv")
pop1 <- ipolate(pop)
growth1 <- ipolate(growth)
growth1[,names(growth1)%in%yrs] = growth1[,names(growth1)%in%yrs]/100 # scaling all values dówn with 100

#merging historical data with population data
popSSP <- merge(mt,pop1,by.x="ISO",by.y="Region")  #merge our data and SSP for population

    
    