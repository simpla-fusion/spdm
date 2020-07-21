# find_package( MDSplus  REQUIRED)
# find_package( Boost  COMPONENTS regex )
# find_package( NetCDF )

add_subdirectory(utility)
add_subdirectory(plugins)

set(SPDB_INCLUDES
  DataBlock.h
  Entry.h  
  EntryInterface.h
  XPath.h
  Document.h
  Collection.h
  DataBase.h
)
 


set(SPDB_SOURCES
  # DataBlock.cpp
  Entry.cpp
  EntryInterface.cpp
  # XPath.cpp
  # Document.cpp
  # Collection.cpp
  # DataBase.cpp
)


add_library(spdb  STATIC  ${SPDB_INCLUDES} ${SPDB_SOURCES} )
target_include_directories(spdb PRIVATE ../extern/)
target_link_libraries(spdb spdb_plugins utility)

install( TARGETS spdb  DESTINATION lib )