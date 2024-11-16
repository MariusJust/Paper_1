######################################## Replication of Burke 2015 #############################################


## cleanup 
cat("\014") 
graphics.off()  # clear all graphs
rm(list = ls()) # remove all files from your workspace
pacman::p_load(pacman, ggplot2, dplyr, lmtest, margins, parallel, maptools, fields, ClassInt, plotrix, sandwich, webuse, tidyr) #installing neccesary packages 
options(scipen = 999, digits = 5)


################################### Part 1 #######################################################
#                                                                                                #                
#                   ESTIMATE GLOBAL RESPONSE WITH BASELINE REGRESSION SPECIFICATION              #
#                                                                                                #
##################################################################################################



# Load the dataset
dataset <- read.csv("Replication/Burke_Data/data/input/GrowthClimateDataset.csv")

# Define the temperature variable
dataset$temp <- dataset$UDel_temp_popweight

# Define the quadratic term for temperature
dataset$temp_sq <- dataset$temp^2

yi_vars <- grep("_yi_", names(dataset), value = TRUE) ## all variables relating to yi
y2_vars <- grep("_y2_", names(dataset), value = TRUE) ## all variables relating to y2

# Combine the variables to include in the regression
all_additional_vars <- c(yi_vars, y2_vars)

# Create the full formula dynamically
# Base formula
formula_base <- as.formula(paste("growthWDI ~ temp + temp_sq + UDel_precip_popweight + UDel_precip_popweight_2 + factor(year) + factor(iso_id)"))

# Full formula including dynamic variables
full_formula <- update(formula_base, paste(". ~ . +", paste(all_additional_vars, collapse = " + ")))

# Run the baseline regression model using temp and temp_sq 
model <- lm(full_formula, data = dataset)

# These results are equivalent to extended data table 1 - base case
summary(model)




#####################  Caluclating predictive margins to replicate figure 2.A ############################



# Define the range of temperatures, as specified in the article
temp_values <- seq(-5, 35, by = 1)

# Create an empty list to store the average predictions for each temperature
average_predictions <- vector("list", length(temp_values))

# Step 1: Create vectors for lower and upper bounds of the confidence intervals
lower_bound <- numeric(length(temp_values))
upper_bound <- numeric(length(temp_values))

# Step 2: Use the data that was used in the model as the "new_data" in the prediction function
new_data <- model$model  # Use the model's data directly

names(new_data)[names(new_data) == "factor(year)"] <- "year"
names(new_data)[names(new_data) == "factor(iso_id)"] <- "iso_id"

# Step 3: Loop over each temperature value
for (i in seq_along(temp_values)) {
  
  temp_val <- temp_values[i]
  
  # Step 3: Fix temp at the current value and adjust the quadratic term for temp
  new_data$temp <- temp_val  
  new_data$temp_sq <- temp_val^2  
  
  # Step 4: Make predictions for the current temp value, including standard errors
  pred <- predict(model, newdata = new_data, se.fit = TRUE)
  
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


