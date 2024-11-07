
source("helper_functions.R")

#Retrieve longitude and latitudes 
unpack(dim(), c("lon", "lat"))

#Retrieve the names of the countries in the model
countries <-unique(read_excel("Data/MainData.1.xlsx", range=cell_cols("B:C"))) 

#Retrieve the coordinates that define the borders for each country
matched_country_shp <- read_sf("Data/national-identifier-grid/gpw_v4_national_identifier_grid_rev11_30_min.shp") %>%
  filter(ISOCODE %in% countries$`Country Code`)

final_country_data <- st_make_valid(left_join(matched_country_shp, countries, by = c("ISOCODE" = "Country Code"))[,c(2,4)])


#################################################################### Creating function to assign country #####################################################




country_grid <- match_grid_to_countries(lon, lat, final_country_data)

# Find the positions in the matrix where the value is "AFG"


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

visualise_coordinates("COD")

lon
write_csv(country_grid, "country_grid.csv")
