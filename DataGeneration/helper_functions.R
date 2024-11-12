
# Function to get a better understanding of the NC file

NC_info <- function(nc_file) {
  # Open the NetCDF file
  
  
  cat("NetCDF File Information:\n")
  
  # Loop through each dimension in the file
  for (dim_name in names(nc_file$dim)) {
    # Extract the dimension
    dim_info <- nc_file$dim[[dim_name]]
    
    # Print the dimension details
    cat("The file has:", dim_info$len, "data points, measuring", dim_info$units, "in the", dim_name, "dimension.\n")
  }
  
}


#Function to retrieve the lat, lon dimensions of the NC files

dim <- function(){
  
  lon <-  read.table("DataGeneration/Data/dim_lon.csv", header = TRUE, sep = ",", 
                     colClasses = c("NULL", "character"))
  
  lon$longitude <- as.numeric(lon$longitude)
  
  lat <-  read.table("DataGeneration/Data/dim_lat.csv", header = TRUE, sep = ",", 
                     colClasses = c("NULL", "numeric"))
  
  lon <- as.numeric(lon[[1]]) 
  lat <-as.numeric(lat[[1]]) 
  
  return(list(lon = lon, lat = lat))
}

# Function that maps each longitude and lattitude grid to a country ISOCODE
match_grid_to_countries <- function(lon_vals, lat_vals, country_shp) {
  # Create a grid of longitude and latitude values (combine them into one table)
  grid_points <- expand.grid(lon = lon_vals, lat = lat_vals)
  
  # Convert the grid points to an sf object (points geometry)
  grid_sf <- st_as_sf(grid_points, coords = c("lon", "lat"), crs = st_crs(country_shp))
  
  # Perform spatial join: match grid points with the country polygons
  joined_sf <- st_join(grid_sf, country_shp, left = TRUE)
  
  # Create a data frame that includes the longitude, latitude, and the country ISOCODE
  result_df <- data.frame(
    lon = st_coordinates(joined_sf)[, 1],  # Extract longitude
    lat = st_coordinates(joined_sf)[, 2],  # Extract latitude
    country = joined_sf$ISOCODE            # Extract the ISOCODE
  )
  
  return(result_df)
}


match_grid_to_countries_hash <- function(lon_vals, lat_vals, country_shp) {
  # Create a grid of longitude and latitude values (combine them into one table)
  grid_points <- expand.grid(lon = lon_vals, lat = lat_vals)
  
  # Convert the grid points to an sf object (points geometry)
  grid_sf <- st_as_sf(grid_points, coords = c("lon", "lat"), crs = st_crs(country_shp))
  
  # Perform spatial join: match grid points with the country polygons
  joined_sf <- st_join(grid_sf, country_shp, left = TRUE)
  
  # Initialize the hashmap
  lon_lat_idx_to_country <- hash()
  
  # Populate the hashmap using the integer indices (lon_idx and lat_idx) instead of real lon/lat values
  for (lon_idx in 1:length(lon_vals)) {
    for (lat_idx in 1:length(lat_vals)) {
      lon <- lon_vals[lon_idx]  # Get the actual longitude
      lat <- lat_vals[lat_idx]  # Get the actual latitude
      
      # Find the corresponding row in the joined_sf using lon and lat values
      matching_row <- which(st_coordinates(joined_sf)[, 1] == lon & st_coordinates(joined_sf)[, 2] == lat)
      
      if (length(matching_row) > 0) {
        country_code <- joined_sf$ISOCODE[matching_row]  # Get the country code
        
        # Create a key using lon_idx and lat_idx
        key <- paste(lon_idx, lat_idx, sep = ",")
        
        # Store the country code in the hashmap
        lon_lat_idx_to_country[[key]] <- country_code
      }
    }
  }
  
  return(lon_lat_idx_to_country)
}



unpack <- function(x, names) {
  for (i in seq_along(names)) {
    assign(names[i], x[[i]], envir = .GlobalEnv)
  }
}

year_month <- function(int){
  
  days_since_1900 <- time_data[int]
  
  # Convert to Date
  date <- as.Date("1900-01-01") + days_since_1900
  
  # Extract the month and year
  year <- format(date, "%Y")
  month <- format(date, "%m")
  
  return (list(month = month, year = year))
}


visualise_coordinates <- function(ISO){
  
  country_shp  <- read_sf("Data/national-identifier-grid/gpw_v4_national_identifier_grid_rev11_30_min.shp")
  
  
  coordinates <- subset(country_grid, country=="DNK")
  
  map <- st_geometry(ne_countries(country=subset(country_shp, ISOCODE==ISO)["NAME0"],
                                  returnclass = "sf"))
  
  ggplot()+geom_sf(data=map)+theme_minimal() +
    # Add points for Germany coordinates
    geom_point(data = coordinates, 
               aes(x = lon, y = lat),  # Replace with actual column names
               color = "red", size = 2)
  
  
}





