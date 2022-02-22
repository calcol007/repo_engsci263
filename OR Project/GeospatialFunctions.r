arrowRoute=function(routeFilename,storeDf) {
  #Identify which kind of map should be shown and put on label
  if(identical(storeDf,supermarket)) {
    name='Weekday '
  } else {
    name='Saturday '
  }
  #Read the route file
  usedRoute=readLines(routeFilename)
  #Create matrix to store the route file
  tripMatrix=matrix(0,nrow(storeDf)-1,4)
  colnames(tripMatrix)=c('Long','Lat','LongEnd','LatEnd')
  #Initialise node reach(the row for the matrix)
  nodeReach=1
  for(i in 1:length(usedRoute)) {
    #Initialise start point and split the string
    route=as.vector(st_coordinates(filter(storeDf,Store=='Distribution Centre Auckland')$geometry))
    used=strsplit(usedRoute[i],split=', ')
    for(j in 1:length(used[[1]])) {
      #Add coordinate to the start and the end point of the arrows
      tripMatrix[,'Long'][nodeReach]=route[1]
      tripMatrix[,'Lat'][nodeReach]=route[2]
      #Add coordinate to the end point of arrows
      route1=as.vector(st_coordinates(filter(storeDf,Store==used[[1]][j])$geometry))
      tripMatrix[,'LongEnd'][nodeReach]=route1[1]
      tripMatrix[,'LatEnd'][nodeReach]=route1[2]
      route=route1
      nodeReach=nodeReach+1
    }
  }
  #change tripMatrix to a dataframe
  tripMatrix=as.data.frame(tripMatrix)
  #Return the map
  map=ggplot(akl_shape)+
    geom_sf(data=akl_shape,fill='#c7c7c7')+
    geom_sf(data=storeDf,aes(color=Type))+
    scale_color_brewer(palette="Dark2")+
    geom_segment(data=tripMatrix,aes(x=Long,y=Lat,xend=LongEnd,yend=LatEnd),
                 arrow=arrow(length=unit(0.2,'cm')),alpha=0.3)+
    labs(title=paste0(name, 'Routes'))
  return(map)
}


regionClassifier=function(routeFilename,storeDf) {
  #Read the route file
  usedRoute=readLines(routeFilename)
  usedRoute=as.data.frame(usedRoute)
  colnames(usedRoute)='Route'
  usedRoute$Region='Unknown'
  for(i in 1:nrow(usedRoute)) {
    #split the string for the route
    route=str_split(usedRoute[i,1],', ')
    usedRoute[i,2]=filter(storeDf,Store==route[[1]][1])$Region
  }
  return(usedRoute)
}


regionalRoute=function (regional_raster,regional_trips,regional_sat_trips,region_stores) {
  #This function plot regional routes for trips to Woolworth stores
  Sat_stores=filter(region_stores,Type%in%c('Countdown','Distribution Centre'))
  distro=filter(region_stores,Type=='Distribution Centre')
  #Plot the base for the regional routes(raster and store location)
  base=tm_shape(regional_raster)+
    tm_rgb()+tm_shape(region_stores)+
    tm_dots(size=0.3,col='Type',palette='Dark2')+
    tm_layout(legend.outside = TRUE)
  base_sat=tm_shape(regional_raster)+
    tm_rgb()+tm_shape(Sat_stores)+
    tm_dots(size=0.3,col='Type',palette='Dark2')+
    tm_layout(legend.outside=TRUE)
  #Get the route to each store in each route of the region, including the route back
  for(i in 1:nrow(regional_trips)) {
    #Split the route into set of Strings
    route=str_split(regional_trips[i,1],', ')
    #Draw the route for each individual trip
    for (j in 1:length(route[[1]])) {
      if(j==1) {
        pointFrom=distro
        pointTo=filter(region_stores,Store==route[[1]][j])
      } else if(j==length(route[[1]])) {
        pointFrom=filter(region_stores,Store==route[[1]][j-1])
        pointTo=distro
      } else {
        pointFrom=filter(region_stores,Store==route[[1]][j-1])
        pointTo=filter(region_stores,Store==route[[1]][j])
      }
      trip=osrmRoute(pointFrom,pointTo)
      trip=st_as_sf(trip,coords=c('lon','lat'),crs=4326)
      triplines=trip%>%summarise(do_union=F)%>%st_cast('LINESTRING')
      base=base+tm_shape(triplines)+tm_lines(col='blue')
    }
  }
  for(i in 1:nrow(regional_sat_trips)) {
    #Split the route into set of Strings
    route=str_split(regional_sat_trips[i,1],', ')
    #Draw the route for each individual trip
    for (j in 1:length(route[[1]])) {
      if(j==1) {
        pointFrom=distro
        pointTo=filter(region_stores,Store==route[[1]][j])
      } else if(j==length(route[[1]])) {
        pointFrom=filter(region_stores,Store==route[[1]][j-1])
        pointTo=distro
      } else {
        pointFrom=filter(region_stores,Store==route[[1]][j-1])
        pointTo=filter(region_stores,Store==route[[1]][j])
      }
      trip_sat=osrmRoute(pointFrom,pointTo)
      trip_sat=st_as_sf(trip_sat,coords=c('lon','lat'),crs=4326)
      triplines_sat=trip_sat%>%summarise(do_union=F)%>%st_cast('LINESTRING')
      base_sat=base_sat+tm_shape(triplines_sat)+tm_lines(col='blue')
    }
  }
  region_map=tmap_arrange(base,base_sat,ncol = 2)
  return (region_map)
}
