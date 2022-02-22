library(spData)
library (tidyverse)
library(tmap)
library(mapedit)
library(OpenStreetMap)
library(raster)
library(rgdal)
library(osrm)
source('GeospatialFunctions.R')
#Read Supermarket location data from the csv
supermarket_raw=read.csv('data/WoolworthsRegion.csv')
#Convert Longitude and Latitude to be a geometry, using EPSG 4326 system
supermarket1=st_as_sf(supermarket_raw,coords=c("Long","Lat"),crs=4326)
#Transform coordinate into NZGD2000 coordinate system
supermarket=st_transform(supermarket1,2193)
#Set tmap mode to plot
tmap_mode("plot")
#Read the shape file for New Zealand for territorial authorities
nz_shape=st_read("Shape_files/TA_2018_clipped")
#Get the shape of Auckland
akl_shape=nz_shape%>%filter(TA2018_V_1=="Auckland")
#Crop the map for the area of interest
akl_shape=st_crop(akl_shape,xmin=1740000,xmax=1781000,ymin=5890000,ymax=5940000)
akl_shape1=st_transform(akl_shape,4326)
#Initially visualise data for weekdays
data_vis=tm_shape(akl_shape)+tm_polygons()+tm_shape(supermarket)+tm_dots(col='Type',palette='Dark2',size=0.3)+ tm_layout(legend.position = c("right", "top"), 
                                                                                                                         title= 'Woolworth supermarkets location', 
                                                                                                                         title.position = c('right', 'top'),legend.frame = TRUE)
#Save the plot above as png
tmap_save(data_vis,'Plot/locations.png')
#Create supermarket data for Saturday(Only Countdowns and the Distribution center included)
saturday=filter(supermarket,Type%in%c('Countdown','Distribution Centre'))
#Initially visualise data for weekdays
data_vis_sat=tm_shape(akl_shape)+tm_polygons()+tm_shape(saturday)+tm_dots(col='Type',palette='Dark2',size=0.3)+tm_layout(legend.position = c("right", "top"), 
                                                                                                                         title= 'Woolworth supermarkets locations (Saturday demands)', 
                                                                                                                         title.position = c('right', 'top'),legend.frame = TRUE)
#Save the plot as png
tmap_save(data_vis_sat,'Plot/Saturday.png')

#Get the coordinate of supermarket in each region and the extent coordinates
South=filter(supermarket1,Region%in%c('South','Distribution Centre Auckland'))
South_shape=st_crop(akl_shape,xmin=1758000,xmax=1777000,ymin=5895000,ymax=5913000)
North=filter(supermarket1,Region%in%c('North','Distribution Centre Auckland'))
North_shape=st_crop(akl_shape,xmin=1751000,xmax=1766000,ymin=5908000,ymax=5936000)
Central=filter(supermarket1,Region%in%c('Central','Distribution Centre Auckland'))
Central_shape=st_crop(akl_shape,xmin=1750000,xmax=1765000,ymin=5907800,ymax=5922000)
East=filter(supermarket1,Region%in%c('East','Distribution Centre Auckland'))
East_shape=st_crop(akl_shape,xmin=1759000,xmax=1773000,ymin=5906800,ymax=5918500)
West=filter(supermarket1,Region%in%c('West','Distribution Centre Auckland'))
West_shape=st_crop(akl_shape,xmin=1741100,xmax=1762000,ymin=5908000,ymax=5928000)

weekdaymap=arrowRoute(routeFilename = 'data/WeekdayRouteStores.txt',storeDf = supermarket)
weekdaymap
ggsave('Plot/WeekdayRoute.png')
#Read the route file for Saturday route and save as a png
satmap=arrowRoute(routeFilename = 'data/SaturdayRouteStores.txt',storeDf= saturday)
satmap
ggsave('Plot/SaturdayRoute.png')
#Transform all the region shape to 4326 to plot the regional route map
North_shape=st_transform(North_shape,4326)
South_shape=st_transform(South_shape,4326)
East_shape=st_transform(East_shape,4326)
West_shape=st_transform(West_shape,4326)
Central_shape=st_transform(Central_shape,4326)
#Classify region for weekdayRoute
w=regionClassifier(routeFilename = 'data/WeekdayRouteStores.txt',storeDf = supermarket)
#Do the same thing for Saturday routes
s=regionClassifier(routeFilename = 'data/SaturdayRouteStores.txt',storeDf = saturday)

#Put each route region into a new data frame
North_route=filter(w,Region=='North')
South_route=filter(w,Region=='South')
East_route=filter(w,Region=='East')
West_route=filter(w,Region=='West')
Central_route=filter(w,Region=='Central')
North_sat_route=filter(s,Region=='North')
South_sat_route=filter(s,Region=='South')
East_sat_route=filter(s,Region=='East')
West_sat_route=filter(s,Region=='West')
Central_sat_route=filter(s,Region=='Central')
#Save the North shape as a raster for reusability
akl_North=openmap(as.numeric(st_bbox(North_shape))[c(4,1)],as.numeric(st_bbox(North_shape))[c(2,3)],type='osm')
akl_North=openproj(akl_North)
North_raster=raster(akl_North)
North_raster=writeRaster(North_raster,'Shape_files/North.tif',format='GTiff',overwrite=TRUE)
#If the file is available in Shape_files, run this command only
North_raster=raster('Shape_files/North.tif')

#Save the South shape as a raster for reusability
akl_South=openmap(as.numeric(st_bbox(South_shape))[c(4,1)],as.numeric(st_bbox(South_shape))[c(2,3)],type='osm')
akl_South=openproj(akl_South)
South_raster=raster(akl_South)
South_raster=writeRaster(South_raster,'Shape_files/South.tif',format='GTiff')
#If the file is available in Shape_files,run this command only
South_raster=raster('Shape_files/South.tif')


#Save the East shape as a raster for reusability
akl_East=openmap(as.numeric(st_bbox(East_shape))[c(4,1)],as.numeric(st_bbox(East_shape))[c(2,3)],type='osm')
akl_East=openproj(akl_East)
East_raster=raster(akl_East)
East_raster=writeRaster(East_raster,'Shape_files/East.tif',format='GTiff')
#If the file is available in Shape_files,run this command only
East_raster=raster('Shape_files/East.tif')

#Save the West shape as a raster for reusability
akl_West=openmap(as.numeric(st_bbox(West_shape))[c(4,1)],as.numeric(st_bbox(West_shape))[c(2,3)],type='osm')
akl_West=openproj(akl_West)
West_raster=raster(akl_West)
West_raster=writeRaster(West_raster,'Shape_files/West.tif',format='GTiff')
#If the file is saved in Shape_files,run this command only
West_raster=raster('Shape_files/West.tif')


#Save the Central shape as a raster for reusability
akl_Central=openmap(as.numeric(st_bbox(Central_shape))[c(4,1)],as.numeric(st_bbox(Central_shape))[c(2,3)],type='osm')
akl_Central=openproj(akl_Central)
Central_raster=raster(akl_Central)
Central_raster=writeRaster(Central_raster,'Shape_files/Central.tif',format='GTiff')
#If the file is saved in Shape_files,run this command only
Central_raster=raster('Shape_files/Central.tif')
#Create the tif file for Auckland map for reusability
akl_map=openmap(as.numeric(st_bbox(akl_shape1))[c(4,1)],as.numeric(st_bbox(akl_shape1))[c(2,3)],type='osm')
akl_map=openproj(akl_map)
akl_raster=raster(akl_map)
akl_raster=writeRaster(akl_raster,'Shape_files/Auckland.tif',format='GTiff')
#If the tif file is available, run this command only
akl_raster=raster('Shape_files/Auckland.tif')

#Plot the map for each region
North_map=regionalRoute(regional_raster= North_raster,regional_trips = North_route,regional_sat_trips = North_sat_route,region_stores = North)
tmap_save(North_map,'Plot/North_routes.png',width = 2100,height=1200)
South_map=regionalRoute(regional_raster= South_raster,regional_trips = South_route,regional_sat_trips = South_sat_route,region_stores = South)
tmap_save(South_map,'Plot/South_routes.png',width = 2500,height=900)
East_map=regionalRoute(regional_raster= East_raster,regional_trips = East_route,regional_sat_trips = East_sat_route,region_stores = East)
tmap_save(East_map,'Plot/East_routes.png',width = 2500,height=850)
West_map=regionalRoute(regional_raster= West_raster,regional_trips = West_route,regional_sat_trips = West_sat_route,region_stores = West)
tmap_save(West_map,'Plot/West_routes.png',width = 2500,height=850)
Central_map=regionalRoute(regional_raster= Central_raster,regional_trips = Central_route,regional_sat_trips = Central_sat_route,region_stores = Central)
tmap_save(Central_map,'Plot/Central_routes.png',width = 2500,height=900)
Auckland_map=regionalRoute(regional_raster= akl_raster,regional_trips = w,regional_sat_trips = s,region_stores = supermarket1)
tmap_save(Auckland_map,'Plot/Auckland_routes.png',width=2500,height=1100)