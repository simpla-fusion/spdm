# find_package( Boost REQUIRED )
# find_package( HDF5 REQUIRED )

include(CTest)
enable_testing()



include_directories(../source)
include_directories(../extern)


 
add_executable(test_node test_node.cpp )
add_test( test_node test_node )

add_executable(test_hdata test_hdata.cpp )
add_test( test_hdata test_hdata )
 
