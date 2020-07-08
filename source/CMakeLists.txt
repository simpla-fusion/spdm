# find_package( MDSplus  REQUIRED)
# find_package( Boost  COMPONENTS regex )
# find_package( NetCDF )


set(SPDB_INCLUDES
  Util.h
  Entry.h
  Node.h
  XPath.h
  
)
 


set(SPDB_SOURCES
  XPath.cpp
  Entry.cpp
  EntryImpl.cpp
  Node.cpp
  Util.cpp
    # ../extern/pugixml/pugixml.cpp
)


add_library(spdb SHARED ${SPDB_INCLUDES} ${SPDB_SOURCES} )
target_include_directories(spdb PRIVATE ../extern/)

install( TARGETS spdb  DESTINATION lib )