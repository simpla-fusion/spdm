# find_package( MDSplus  REQUIRED)
# find_package( Boost  COMPONENTS regex )
# find_package( NetCDF )


set(SPDB_INCLUDES
  DataBlock.h
  Node.h
  Entry.h  
  Document.h
  Schema.h
  XPath.h
  Collection.h
  DataBase.h
)
 


set(SPDB_SOURCES
  Node.cpp
  NodePlugin.cpp
  Entry.cpp  
  DataBlock.cpp
  XPath.cpp
  Document.cpp
  Schema.cpp
  # Collection.cpp
  # DataBase.cpp
)


add_library(spdb  STATIC  ${SPDB_INCLUDES} ${SPDB_SOURCES} )
target_include_directories(spdb PRIVATE ../extern/)
target_link_libraries(spdb utility)

install( TARGETS spdb  DESTINATION lib )