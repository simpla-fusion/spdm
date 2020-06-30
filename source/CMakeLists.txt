# find_package( MDSplus  REQUIRED)
# find_package( Boost  COMPONENTS regex )
# find_package( NetCDF )


set(SPDB_INCLUDES
  # SpDBObject.h
  # SpUtil.h
  # SpDB.h
)
 


set(SPDB_SOURCES
  # SpDB.cpp
  SpDocument.cpp
  SpUtil.cpp
    # ../extern/pugixml/pugixml.cpp
)


add_library(spdb SHARED ${SPDB_INCLUDES} ${SPDB_SOURCES} )
target_include_directories(spdb PRIVATE ../extern/)

install( TARGETS spdb  DESTINATION lib )