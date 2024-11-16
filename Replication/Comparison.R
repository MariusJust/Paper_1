######################################## Comparison #############################################


## cleanup 
cat("\014") 
graphics.off()  # clear all graphs
rm(list = ls()) # remove all files from your workspace
pacman::p_load(pacman, ggplot2, dplyr, lmtest, margins, parallel, maptools, fields, ClassInt, plotrix, sandwich, webuse, tidyr, readxl, stargazer,texreg) #installing neccesary packages 
options(scipen = 999, digits = 5)


################################### Part 1 #######################################################
#                                                                                                #                
#                         Load original and updated Data                                         #
#                                                                                                #
##################################################################################################

# Load the dataset
Original_Data <- read.csv("Replication/GrowthClimateDataset.csv")

# Load the updated dataset
dataset <- read_excel("MainData.xlsx")

# Define the range of temperatures, as specified in the article
temp_values <- seq(-5, 35, by = 1)


################################### Part 2 #######################################################
#                                                                                                #                
#                              Original Model                                                    #
#                                                                                                #
##################################################################################################


# Define the temperature variable
Original_Data$temp <- Original_Data$UDel_temp_popweight

# Define the quadratic term for temperature
Original_Data$temp_sq <- Original_Data$temp^2


Original_Data$precip<- Original_Data$UDel_precip_popweight

Original_Data$precip_sq <- Original_Data$precip^2

yi_vars <- grep("_yi_", names(Original_Data), value = TRUE) ## all variables relating to yi
y2_vars <- grep("_y2_", names(Original_Data), value = TRUE) ## all variables relating to y2

# Combine the variables to include in the regression
all_additional_vars <- c(yi_vars, y2_vars)

# Create the full formula dynamically
# Base formula
formula_base <- as.formula(paste("growthWDI ~ temp + temp_sq + precip + precip_sq + factor(year) + factor(iso_id)"))

# Full formula including dynamic variables
full_formula <- update(formula_base, paste(". ~ . +", paste(all_additional_vars, collapse = " + ")))

# Run the baseline regression model using temp and temp_sq 
model <- lm(full_formula, data = Original_Data)


#####################  Caluclating predictive margins for the original data ############################


# Create an empty list to store the average predictions for each temperature
average_predictions <- vector("list", length(temp_values))

# Step 1: Create vectors for lower and upper bounds of the confidence intervals
lower_bound <- numeric(length(temp_values))
upper_bound <- numeric(length(temp_values))

# Step 2: Use the data that was used in the model as the "new_data" in the prediction function
old_data <- model$model  # Use the model's data directly

names(old_data)[names(old_data) == "factor(year)"] <- "year"
names(old_data)[names(old_data) == "factor(iso_id)"] <- "iso_id"

# Step 3: Loop over each temperature value
for (i in seq_along(temp_values)) {
  
  temp_val <- temp_values[i]
  
  # Step 3: Fix temp at the current value and adjust the quadratic term for temp
  old_data$temp <- temp_val  
  old_data$temp_sq <- temp_val^2  
  
  # Step 4: Make predictions for the current temp value, including standard errors
  pred <- predict(model, newdata = old_data, se.fit = TRUE)
  
  # Step 5: Calculate the average prediction and 90% confidence interval
  average_predictions[i] <- mean(pred$fit, na.rm = TRUE)
  
  # 90% confidence interval (z-value for 90% CI is approximately 1.645)
  # ci_multiplier <- 1.645
  # lower_bound[i] <- mean(pred$fit - ci_multiplier * pred$se.fit, na.rm = TRUE)
  # upper_bound[i] <- mean(pred$fit + ci_multiplier * pred$se.fit, na.rm = TRUE)
}

# Step 4: Combine the results into a data frame
results_old <- data.frame(
  temp = temp_values,
  avg_prediction = unlist(average_predictions)
  # lower_bound = lower_bound,
  # upper_bound = upper_bound
)






################################### Part 3 #######################################################
#                                                                                                #                
#                                 Updated Data                                                   #
#                                                                                                #
##################################################################################################


# Define the temperature variable 
dataset$temp <- dataset$TempPopWeight
# Define the quadratic term for temperature
dataset$temp_sq <- dataset$temp^2
dataset$precip<-dataset$PrecipPopWeight
dataset$precip_sq<-dataset$precip^2

up_yi_vars <- grep("_yi_", names(dataset), value = TRUE) ## all variables relating to yi
up_y2_vars <- grep("_y2_", names(dataset), value = TRUE) ## all variables relating to y2

# Combine the variables to include in the regression
up_all_additional_vars <- c(up_yi_vars, up_y2_vars)

# Create the full formula dynamically
# Base formula
up_formula_base <- as.formula(paste("GrowthWDI~ temp + temp_sq + precip + precip_sq + factor(Year) + factor(ISO)"))

# Full formula including dynamic variables
up_full_formula <- update(up_formula_base, paste(". ~ . +", paste(up_all_additional_vars, collapse = " + ")))


# Run the baseline regression model using temp and temp_sq 
model_new <- lm(up_full_formula, data = dataset)

# Create an empty list to store the average predictions for each temperature
average_predictions <- vector("list", length(temp_values))

# Step 1: Create vectors for lower and upper bounds of the confidence intervals
lower_bound <- numeric(length(temp_values))
upper_bound <- numeric(length(temp_values))

# Step 2: Use the data that was used in the model as the "new_data" in the prediction function
new_data <- model_new$model  # Use the model's data directly

colnames(new_data)[6] <- "Year"
colnames(new_data)[7] <- "ISO"


#####################  Caluclating predictive margins for the updated data ############################

# Step 3: Loop over each temperature value
for (i in seq_along(temp_values)) {
  
  temp_val <- temp_values[i]
  
  # Step 3: Fix temp at the current value and adjust the quadratic term for temp
  new_data$temp <- temp_val  
  new_data$temp_sq <- temp_val^2  
  
  # Step 4: Make predictions for the current temp value, including standard errors
  pred <- predict(model_new, newdata = new_data, se.fit = TRUE)
  
  # Step 5: Calculate the average prediction and 90% confidence interval
  average_predictions[i] <- mean(pred$fit, na.rm = TRUE)
  
  # 90% confidence interval (z-value for 90% CI is approximately 1.645)
  ci_multiplier <- 1.645
  # lower_bound[i] <- mean(pred$fit - ci_multiplier * pred$se.fit, na.rm = TRUE)
  # upper_bound[i] <- mean(pred$fit + ci_multiplier * pred$se.fit, na.rm = TRUE)
}

# Step 4: Combine the results into a data frame
results <- data.frame(
  temp = temp_values,
  avg_prediction = unlist(average_predictions)
  # lower_bound = lower_bound,
  # upper_bound = upper_bound
)




################################### Part 4 #######################################################
#                                                                                                #                
#                             Plotting the predictive margins                                    #
#                                                                                                #
##################################################################################################


# Add a column to each dataframe to identify them
results$source <- "New Model"     # Label for the new model results
results_old$source <- "Old Model" # Label for the old model results
#results_new_old$source <- "New Model, old filter" # Label for the old model results

# Combine the two dataframes
combined_results <- rbind(results, results_old)

# Plot both models in the same graph
ggplot(combined_results, aes(x = temp, y = avg_prediction, color = source)) +
  geom_line() +  # Plot the predictions for both models
  
  labs(title = "Predicted Growth with 90% Confidence Interval",
       x = "Temperature",
       y = "Average Predicted Growth",
       color = "Model", fill = "Model") +  # Adjust the legend labels
  theme_minimal()

###################################################

#Now I change the updated data so it contains the same year values and country values as in the original data














