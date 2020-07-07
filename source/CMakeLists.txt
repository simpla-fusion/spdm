# find_package( MDSplus  REQUIRED)
# find_package( Boost  COMPONENTS regex )
# find_package( NetCDF )


set(SPDB_INCLUDES
  SpUtil.h
  SpNode.h
  
)
 


set(SPDB_SOURCES
  SpNode.cpp
  
  SpUtil.cpp
    # ../extern/pugixml/pugixml.cpp
)


add_library(spdb SHARED ${SPDB_INCLUDES} ${SPDB_SOURCES} )
target_include_directories(spdb PRIVATE ../extern/)

install( TARGETS spdb  DESTINATION lib )