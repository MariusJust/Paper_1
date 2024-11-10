
##################################################################################################
#                                                                                                #                
#                               Main Script                                                      #
#                                                                                                #
##################################################################################################

### Cleanup and loading packages

cat("\014") 
graphics.off()  # clear all graphs
rm(list = ls()) # remove all files from your workspace
pacman::p_load(raster, sf, ncdf4, readxl, dplyr, rnaturalearth,tidyverse, hash, parallel, foreach, doParallel, snow, utils, data.table, progressr) #installing necessary packages 

### Initialising data and loading functions

source("helper_functions.R")
source("Create_Country_Grid.R")

#Retrieve longitude and latitudes 
unpack(dim(), c("lon", "lat"))

countries <-unique(read_excel("Data/MainData.1.xlsx", range=cell_cols("B:C"))) 


#Retrieve the coordinates that define the borders for each country
matched_country_shp <- read_sf("Data/national-identifier-grid/gpw_v4_national_identifier_grid_rev11_30_min.shp") %>%
  filter(ISOCODE %in% countries$`Country Code`)

final_country_data <- st_make_valid(left_join(matched_country_shp, countries, by = c("ISOCODE" = "Country Code"))[,c(2,4)])


####################################################  Loading Data ######################################################
#Dataframe mapping each gridpoint to a country
country_grid <- read_csv("Data/country_grid.csv")

#the NC file, containing population densities for each grid cell
nc_file_density <- nc_open("Data/gpw_v4_population_density_rev11_30_min.nc")

#Information about the dimensions of the density file 
NC_info(nc_file_density)

#The 1'st raster corresponds to population density in year 2000. This can be seen through the csv file that is available upon download. 
baseline_density_year <- 1


#retrieve data for the year 2000 only
data_density <- ncvar_get(nc_file_density, start = c(1, 1, baseline_density_year), count = c(-1, -1, 1))

density_lon <-  nc_file_density$dim$longitude$vals
density_lat <- nc_file_density$dim$latitude$vals


##################################################################################################
#                                                                                                #                
#             Calculating density-weigthed precipitation averages by country                     #
#                                                                                                #
##################################################################################################

nc_file_precip <- nc_open("Data/cru_ts4.08.1901.2023.pre.dat.nc")
nc_file_temp <- nc_open("Data/cru_ts4.08.1901.2023.tmp.dat.nc")

#filter data based on the year 1960
time_data <- ncvar_get(nc_file_precip, varid="time")
start_index <- which(time_data>= 365 * 60)[1] # 60 is the number of years after 1900
total_time_points<- length(time_data)-start_index+1

data_precip <- ncvar_get(nc_file_precip,
                    "pre", start = c(1, 1, start_index), count = c(-1, -1, total_time_points))

data_temp <- ncvar_get(nc_file_temp,
                       "tmp", start = c(1, 1, start_index), count = c(-1, -1, total_time_points))


start_year <- 1960
end_year <- 2023

precip_lon <-  nc_file_precip$dim$lon$vals
precip_lat <- nc_file_precip$dim$lat$vals

# --- Create yearly and monthly sequences efficiently
yearly_sequence <- seq(start_year, end_year, 1)

# Create a monthly repetition using a vectorized approach
monthly_rep <- rep(1:12, times = length(yearly_sequence))

# Create a yearly repetition using a vectorized approach
yearly_repetition <- rep(yearly_sequence, each = 12)

num_countries <- nrow(countries[,1])

yearly_rep <- rep(yearly_repetition, times=num_countries)

monthly_rep <- rep(monthly_rep, times = num_countries)

# Repeat the country names for each year-month combination
country_rep <- rep(countries$`Country Code`, each = length(yearly_repetition))

year_month<-data.frame(
  year=yearly_repetition,
  month=monthly_rep
  
)



# --- Create  storage data frame ('out_df'): Column1: Year; Column2: Month; Column 3: Country; Column 4: Precip
out_df <- data.frame(
  year = yearly_rep,
  month = monthly_rep,
  country = country_rep,
  precipitation = 0, 
  temp=0
  
)



##################################################################################################
#                                                                                                #                
#                               Pre computations                                                 #
#                                                                                                #
##################################################################################################

cat("Precomputing valid grid points and sorting keys for all time steps...\n")





# Initialize the hashmap
lon_lat_to_country <- hash()

# Loop through the rows of country_grid and add entries to the hashmap
for (i in 1:nrow(country_grid)) {
  # Create a unique key for each combination of lon and lat
  lon_lat_key <- paste(country_grid$lon[i], country_grid$lat[i], sep = ",")
  
  # Add the corresponding country to the hashmap using the lon-lat key
  lon_lat_to_country[[lon_lat_key]] <- country_grid$country[i]
}

# Now, lon_lat_to_country contains mappings from lon-lat to the country


# Initialize a nested hashmap to store valid grid points for all time steps
valid_grid_points_all_t <- hash()


for (t in 1:768) {
  
  # Initialize a new hashmap for the current time step
  valid_grid_points <- hash()
  cat("starting iteration:",t, "\n")
  for (lon_idx in 1:720) {
    
    valid_lat_idx <- NULL  # Initialize an empty matrix to store valid latitude indices
    
    # Find valid latitude indices for this longitude and time step
    for (lat_idx in 1:length(density_lat)) {
      corresponding_lat_idx <- which(precip_lat == density_lat[lat_idx])
      
      if (length(corresponding_lat_idx) == 1) {
        if (!is.na(data_density[lon_idx, lat_idx]) & !is.na(data_precip[lon_idx, corresponding_lat_idx, t])) {
          valid_lat_idx <- rbind(valid_lat_idx, c(lat_idx, corresponding_lat_idx))  # Append valid latitudes
        }
      }
    }
    
    # Store valid latitude indices for this longitude in the hashmap
    if (length(valid_lat_idx) > 0) {
      valid_grid_points[[as.character(lon_idx)]] <- valid_lat_idx  # Store in the current time step hashmap
    }
  }
  
  # Store the valid grid points hashmap for this time step in the outer hashmap
  valid_grid_points_all_t[[as.character(t)]] <- valid_grid_points
  
}



cat("Completed precomputing valid grid points and sorting keys for all time steps.\n")





##################################################################################################
#                                                                                                #                
#                               Main loop                                                        #
#                                                                                                #
##################################################################################################

# Register all but one core for parallel processing
cores <- detectCores() - 2
cl <- makeCluster(cores, outfile="")
registerDoParallel(cl)



writeLines(c(""), "log.txt")



results <- foreach(t = 1:768, .packages = c("foreach", "doParallel")) %dopar% {
  
  unpack(year_month[t,], c("year","month"))
  
  sink("log.txt", append = TRUE)
  cat("Starting time itteration:", t, "\n")
  sink()
  
  valid_grid_points <- valid_grid_points_all_t[[as.character(t)]]
  
  temp_out_df <- out_df[out_df$year == year & out_df$month == month, ]

  for (lon_idx in 1:720) {
    
    # Get valid latitude indices for the current longitude and time step
    valid_lat_idx <- valid_grid_points[[as.character(lon_idx)]]
    
    
    
    for (int in 1:nrow(valid_lat_idx)) {
      
      lat_idx <- valid_lat_idx[int,2]
      lat_idx_in_density <- valid_lat_idx[int,1]
      
      
      # Retrieve the country code from the hashmap
      country_code <- lon_lat_to_country[[paste(lon[lon_idx], lat[lat_idx], sep = ",")]]
      
   
    
      # cat("Processing longitude index:", lon_idx,
      #     "latitude index in density:", lat_idx_in_density,
      #     "true lattitude is:", density_lat[lat_idx_in_density],
      #     "latitude index in precip:", lat_idx,
      #     "true latitude is:", precip_lat[lat_idx],
      #     "Country code:", country_code, "\n")
      
      if(!is.na(country_code)){
        grid_density <- data_density[lon_idx, lat_idx_in_density]
        grid_precip <- data_precip[lon_idx, lat_idx, t]
        grid_temp <- data_temp[lon_idx, lat_idx, t]
        
        # Debugging cat statements for grid values
        # cat("Grid density at (lon_idx =", lon_idx, ", lat_idx_in_density =", lat_idx_in_density, ") =", grid_density, "\n")
        # cat("Grid precip at (lon_idx =", lon_idx, ", lat_idx =", lat_idx, ") =", grid_precip, "\n")
        # 
        
        country_row <- temp_out_df$year == as.numeric(year) & 
          temp_out_df$month == as.numeric(month) & 
          temp_out_df$country == country_code
        
        temp_out_df$precipitation[country_row] <- temp_out_df$precipitation[country_row] + grid_density * grid_precip
        temp_out_df$temp[country_row] <- temp_out_df$temp[country_row] + grid_density * grid_temp
        # cat("Updated precipitation for country", country_code, ":", updated_precip, "\n")
      }
    }
    
  }
  
  
  return(temp_out_df)  # Return the updated temp_out_df for this iteration
  
}

stopCluster(cl)

cat("All time iterations completed.\n")


# Step 1: Combine the data frames from the list 'results' into one data frame
final_out_df <- do.call(rbind, results)

# Step 2: Sort the data frame by 'country', then 'year', then 'month'
final_out_df <- final_out_df[order(final_out_df$country), ]


for (i in 1:nrow(final_out_df)) {
  # Find the row in out_df where year, month, and country match
  out_df[out_df$year == final_out_df$year[i] &
           out_df$month == final_out_df$month[i] &
           out_df$country == final_out_df$country[i], "precipitation"] <- final_out_df$precipitation[i]
  
  out_df[out_df$year == final_out_df$year[i] &
           out_df$month == final_out_df$month[i] &
           out_df$country == final_out_df$country[i], "temp"] <- final_out_df$temp[i]
}






write.csv2(out_df, "output.csv")



##################################################################################################
#                                                                                                #                
#                              Aggregating and making final output                               #
#                                                                                                #
##################################################################################################


total_densities <- read_csv("~/PhD/Papers/1/Repository/DataGeneration/TotalDensities.csv")

# Ensure column names match for the join
names(total_densities) <- c("country", "total_density")

# Join total densities to the out_df data frame
out_df <- out_df %>%
  left_join(total_densities, by = "country")


yearly_data <- out_df %>%
  group_by(country, year) %>%
  summarize(
    yearly_precipitation = sum(precipitation, na.rm = TRUE)/unique(total_density),  # Summing up the precipitation for each year-country combo
    yearly_temp=sum(temp, na.rm=TRUE )/unique(total_density)/12
    
  )

write.csv2(yearly_data, "final_output.csv")

