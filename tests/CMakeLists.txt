# find_package( Boost REQUIRED )
# find_package( HDF5 REQUIRED )

include(CTest)
enable_testing()



include_directories(../source)
include_directories(../extern)


 
add_executable(test_node test_node.cpp  )
add_test( test_node test_node )
target_link_libraries(test_node utility spdb )

add_executable(test_hdata test_hdata.cpp )
add_test( test_hdata test_hdata )
target_link_libraries(test_hdata  utility  )

add_executable(test_cursor test_cursor.cpp )
add_test( test_cursor test_cursor )
target_link_libraries(test_cursor  utility  )

