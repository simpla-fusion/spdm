# find_package( MDSplus  REQUIRED)
# find_package( Boost  COMPONENTS regex )
# find_package( NetCDF )

add_subdirectory(utility)

set(SPDB_INCLUDES
  Entry.h  
  Node.h
  Attributes.h
  XPath.h
  
  
)
 


set(SPDB_SOURCES
  XPath.cpp
  Node.cpp
  Entry.cpp
  Attributes.cpp
)


add_library(spdb  ${SPDB_INCLUDES} ${SPDB_SOURCES} )
target_include_directories(spdb PRIVATE ../extern/)
target_link_libraries(spdb utility)

install( TARGETS spdb  DESTINATION lib )