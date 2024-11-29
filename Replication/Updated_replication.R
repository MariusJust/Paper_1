######################################## Replication of Burke 2015 #############################################


## cleanup 
cat("\014") 
graphics.off()  # clear all graphs
rm(list = ls()) # remove all files from your workspace
pacman::p_load(pacman, ggplot2, dplyr, lmtest, margins, parallel, maptools, fields, ClassInt, plotrix, sandwich, webuse, tidyr, readxl, RColorBrewer) #installing neccesary packages 
options(scipen = 999, digits = 5)


################################### Part 1 #######################################################
#                                                                                                #                
#                   ESTIMATE GLOBAL RESPONSE WITH BASELINE REGRESSION SPECIFICATION              #
#                                                                                                #
##################################################################################################

# Load the updated dataset
dataset <- read_excel("MainData.xlsx")


# Define the temperature variable
dataset$temp <- dataset$Temp_PopWeight
# Define the quadratic term for temperature
dataset$temp_sq <- dataset$temp^2

dataset$precip<-dataset$Precip_PopWeight
dataset$precip_sq<-dataset$precip^2

yi_vars <- grep("_yi_", names(dataset), value = TRUE) ## all variables relating to time_trend
y2_vars <- grep("_y2_", names(dataset), value = TRUE) ## all variables relating to squared time_trend

# Combine the variables to include in the regression
all_additional_vars <- c(yi_vars, y2_vars)





#####################  Caluclating predictive margins to replicate figure 2.A in Burke ############################
#defining the formula
full_formula <- update(growthWDI~ temp + temp_sq + precip + precip_sq + factor(Year) + factor(ISO), paste(". ~ . +", paste(all_additional_vars, collapse = " + ")))

# Run the baseline regression model using temp and temp_sq 
model_Burke <- lm( full_formula, data = dataset)

# Define the range of temperatures, as specified in the article
temp_values <- seq(-5, 35, by = 1)

# Create an empty list to store the average predictions for each temperature
average_predictions <- vector("list", length(temp_values))

# Step 1: Create vectors for lower and upper bounds of the confidence intervals
lower_bound <- numeric(length(temp_values))
upper_bound <- numeric(length(temp_values))

# Step 2: Use the data that was used in the model as the "new_data" in the prediction function
new_data <- model_Burke$model  # Use the model's data directly

colnames(new_data)[6] <- "Year"
colnames(new_data)[7] <- "ISO"


# Step 3: Loop over each temperature value
for (i in seq_along(temp_values)) {
  
  temp_val <- temp_values[i]
  
  # Step 3: Fix temp at the current value and adjust the quadratic term for temp
  new_data$temp <- temp_val  
  new_data$temp_sq <- temp_val^2  
  
  # Step 4: Make predictions for the current temp value, including standard errors
  pred <- predict(model_Burke, newdata = new_data, se.fit = TRUE)
  
  # Step 5: Calculate the average prediction and 90% confidence interval
  average_predictions[i] <- mean(pred$fit, na.rm = TRUE)
  
  # 90% confidence interval (z-value for 90% CI is approximately 1.645)
  ci_multiplier <- 1.645
  lower_bound[i] <- mean(pred$fit - ci_multiplier * pred$se.fit, na.rm = TRUE)
  upper_bound[i] <- mean(pred$fit + ci_multiplier * pred$se.fit, na.rm = TRUE)
}

# Step 4: Combine the results into a data frame
results <- data.frame(
  temp = temp_values,
  avg_prediction = unlist(average_predictions),
  lower_bound = lower_bound,
  upper_bound = upper_bound
)


#Step 5: Plot
ggplot(results, aes(x = temp, y = avg_prediction)) +
  geom_line(color = "blue") +  # Plot the predictions
  geom_ribbon(aes(ymin = lower_bound, ymax = upper_bound), alpha = 0.2) +  # Add confidence interval
  labs(title = "Predicted Growth with 90% Confidence Interval",
       x = "Temperature",
       y = "Average Predicted Growth") +
  theme_minimal()



#####################  Caluclating predictive margins for 10 precipitation quantiles, as in Leirvik ############################


dataset$precip<-dataset$Precip_PopWeight/1000
dataset$precip_sq<-dataset$precip^2

full_formula <- update(growthWDI ~ temp + temp_sq + precip + precip_sq + 
                         precip:temp + precip:temp_sq + precip_sq:temp + precip_sq:temp_sq + 
                         factor(Year) + factor(ISO), paste(". ~ . +", paste(all_additional_vars, collapse = " + ")))


model_Leirvik <- lm(
 full_formula,
  data = dataset
)


#Only showing summary statistics for variables of interest
summary(model_Leirvik)$coefficients[c("temp", "temp_sq", "precip", "precip_sq", "temp:precip", "temp_sq:precip", "temp:precip_sq", "temp_sq:precip_sq"),]

# Step 1: Define the range of temperatures, as before
temp_values <- seq(-5, 35, by = 1)


# Step 2: Use the data that was used in the model as the "new_data" in the prediction function
new_data <- model_Leirvik$model  # Use the model's data directly
colnames(new_data)[6] <- "Year"
colnames(new_data)[7] <- "ISO"

# Step 1: Compute the 10 deciles of precipitation from the data
precip_deciles <- quantile(dataset$precip, probs = seq(0.1, 1, by = 0.1), na.rm="TRUE")

# Step 2: Create empty lists to store results
all_predictions <- list()  # List to store predictions for each decile


# Step 3: Loop over each precipitation decile (ranked 1 to 10)
for (decile_rank in 1:9) {
  
  # Get the actual precipitation value corresponding to the current decile
  decile_value <- precip_deciles[decile_rank]
  
  # Create vectors to store the average predictions, lower and upper bounds
  avg_pred_decile <- numeric(length(temp_values))
  
  # Step 4: Loop over each temperature value
  for (i in seq_along(temp_values)) {
    
    temp_val <- temp_values[i]
    
    # Fix temp at the current value and precip at the current decile value
    new_data$temp <- temp_val
    new_data$temp_sq <- temp_val^2
    new_data$precip <- decile_value
    new_data$precip_sq <- decile_value^2 
    
    # Step 5: Make predictions for the current temp and precip decile, including standard errors
    pred <- predict(model_Leirvik, newdata = new_data, se.fit = TRUE)
    
    # Step 6: Calculate the average prediction and 90% confidence interval
    avg_pred_decile[i] <- mean(pred$fit, na.rm = TRUE)
    
  }
  
  # Combine predictions into a data frame for the current decile
  results_decile <- data.frame(
    temp = temp_values,
    avg_prediction = avg_pred_decile,
    precip_decile = rep(paste0("Decile:: ", decile_rank), length(temp_values))  # Use decile rank as the label
  )
  
  # Append the results for the current decile to the list of all predictions
  all_predictions[[length(all_predictions) + 1]] <- results_decile
}

# Step 7: Combine all predictions into one data frame
final_results <- do.call(rbind, all_predictions)

# Step 8: Plot the 10 curves in one plot, with decile rank as the label
ggplot(final_results, aes(x = temp, y = avg_prediction, color = factor(precip_decile))) +
  geom_line() +  # Plot the predictions for each decile
  labs(title = "Predicted Growth by Temperature for Different Precipitation Deciles",
       x = "Temperature",
       y = "Average Predicted Growth",
       color = "Precipitation decile") +  # Label for the legend,
      xlim(0,35)  
  scale_color_viridis_d(option = "plasma") +  # Use a color scale to differentiate deciles
  theme_minimal()


  
  
  
  
  
  
  
  
  
  
  
  

  