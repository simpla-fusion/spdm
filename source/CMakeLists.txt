# find_package( MDSplus  REQUIRED)
# find_package( Boost  COMPONENTS regex )
# find_package( NetCDF )

add_subdirectory(utility)

set(SPDB_INCLUDES
  Entry.h  
  EntryInterface.h
  # Attributes.h
  # XPath.h
)
 


set(SPDB_SOURCES
  # XPath.cpp
  Entry.cpp
  EntryInterface.cpp
  EntryMemory.cpp
  # EntryXML.cpp
  # Attributes.cpp
)


add_library(spdb_  STATIC  ${SPDB_INCLUDES} ${SPDB_SOURCES} )
target_include_directories(spdb_ PRIVATE ../extern/)
add_library(spdb INTERFACE)
set_property(TARGET spdb  PROPERTY INTERFACE_LINK_LIBRARIES   -Wl,--whole-archive utility $<TARGET_FILE:spdb_> -Wl,--no-whole-archive) 

install( TARGETS spdb  DESTINATION lib )