# find_package( Boost REQUIRED )
# find_package( HDF5 REQUIRED )

include(CTest)
enable_testing()



include_directories(../source)
include_directories(../extern)


 
add_executable(test_entry test_entry.cpp  )
add_test( test_entry test_entry )
target_link_libraries(test_entry utility spdb )

add_executable(test_hdata test_hdata.cpp )
add_test( test_hdata test_hdata )
target_link_libraries(test_hdata  utility  )

add_executable(test_cursor test_cursor.cpp )
add_test( test_cursor test_cursor )
target_link_libraries(test_cursor  utility  )

add_executable(test_xml test_xml.cpp  ../source/plugins/PluginXML.cpp ../extern/pugixml/pugixml.cpp )
add_test(test_xml test_xml )
target_link_libraries(test_xml utility spdb  )

add_executable(test_hdf5 test_hdf5.cpp  ../source/plugins/EntryHDF5.cpp   )
add_test(test_hdf5 test_hdf5 )
target_link_libraries(test_hdf5 utility spdb  )

add_executable(test_json test_json.cpp  ../source/plugins/EntryJSON.cpp   )
add_test(test_json test_json )
target_link_libraries(test_json utility spdb  )


add_executable(test_yaml test_yaml.cpp  ../source/plugins/EntryYAML.cpp   )
add_test(test_yaml test_yaml )
target_link_libraries(test_yaml utility spdb  )

add_executable(test_path test_path.cpp    )
