# find_package( MDSplus  REQUIRED)
# find_package( Boost  COMPONENTS regex )
# find_package( NetCDF )



set(SPDB_PLUGINS_SOURCES
  EntryMemory.cpp
  EntryXML.cpp
  EntryJSON.cpp
  EntryHDF5.cpp
  ../../extern/pugixml/pugixml.cpp

  # Attributes.cpp
)




add_library(spdb_plugins_  STATIC   ${SPDB_PLUGINS_SOURCES} )
target_include_directories(spdb_plugins_ PRIVATE ${CMAKE_SOURCE_DIR}/extern/)
add_library(spdb_plugins INTERFACE)
set_property(TARGET spdb_plugins  PROPERTY INTERFACE_LINK_LIBRARIES   -Wl,--whole-archive  $<TARGET_FILE:spdb_plugins_> -Wl,--no-whole-archive) 

