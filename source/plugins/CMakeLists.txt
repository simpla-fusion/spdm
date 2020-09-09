

set(SPDB_PLUGINS_SOURCES
  PluginProxy.cpp
)


IF(BUILD_PLUGINS)
  # find_package(PugiXML)
  # if(PugiXML_FOUND)
    set(SPDB_PLUGINS_SOURCES ${SPDB_PLUGINS_SOURCES}  PluginXML.cpp ../../extern/pugixml/pugixml.cpp )
  # endif(PugiXML_FOUND)

  find_package( HDF5 )
  if(HDF5_FOUND)
    set(SPDB_PLUGINS_SOURCES ${SPDB_PLUGINS_SOURCES}  PluginHDF5.cpp)
  endif(HDF5_FOUND)


  find_package( NetCDF )
  if(NetCDF_FOUND)
    set(SPDB_PLUGINS_SOURCES ${SPDB_PLUGINS_SOURCES}  PluginNetCDF.cpp)
  endif(NetCDF_FOUND)

  find_package( MDSplus )
  if(MDSplus_FOUND)
    set(SPDB_PLUGINS_SOURCES ${SPDB_PLUGINS_SOURCES}  PluginMDSplus.cpp)
  endif(MDSplus_FOUND)
ENDIF(BUILD_PLUGINS)

add_library(spdb_plugins_  STATIC   ${SPDB_PLUGINS_SOURCES} )

target_include_directories(spdb_plugins_ PRIVATE ${CMAKE_SOURCE_DIR}/extern/)

add_library(spdb_plugins INTERFACE)

set_property(TARGET spdb_plugins  PROPERTY INTERFACE_LINK_LIBRARIES   -Wl,--whole-archive  $<TARGET_FILE:spdb_plugins_> -Wl,--no-whole-archive) 

