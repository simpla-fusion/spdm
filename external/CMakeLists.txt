#   This directory contains all extern libraries.
#
#
#    TODO: fetch all extern libraries:
#           pugixml
#           Cache2

# find_package(GTest QUIET)

# SET(GTEST_VERSION  release-1.10.0)

# if(NOT GTEST_FOUND)
#  
#   FetchContent_Declare(googletest
#       GIT_REPOSITORY      https://github.com/google/googletest.git
#       GIT_TAG             ${GTEST_VERSION})
#   FetchContent_MakeAvailable(googletest) 

 
#   FetchContent_GetProperties(googletest)
#   if(NOT googletest_POPULATED)
#     FetchContent_Populate(googletest)
#   endif()
 
# endif()

# include(GoogleTest)



# set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/catch)
# add_library(Catch2::Catch IMPORTED INTERFACE)
# set_property(TARGET Catch2::Catch PROPERTY  INTERFACE_INCLUDE_DIRECTORIES "${catch_SOURCE_DIR}")
# set( TESTS
#   test_east_wall 
# )

# foreach( TEST ${TESTS} )
#   BUILD_TEST( ${TEST} ${TEST}.cpp uda_plugins east_plugin )
# endforeach()

# configure_file(
#   "${CMAKE_CURRENT_SOURCE_DIR}/setup.inc.in"
#   "${CMAKE_CURRENT_BINARY_DIR}/setup.inc"
#   @ONLY
# )

# set(NAME east_demo)

# include_directories( ${UDA_CPP_INCLUDE_DIRS} ${UDA_FAT_CPP_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR} )
# link_directories(${UDA_CPP_LIBRARY_DIRS} ${UDA_LIBRARY_DIRS}  ${UDA_FAT_CPP_LIBRARY_DIRS} )

# add_executable( ${NAME} ${NAME}.cpp )

# target_link_libraries( ${NAME} PRIVATE
#     ${UDA_FAT_CPP_LIBRARIES}
#     ${OPENSSL_LIBRARIES}
#     # test_helpers
#     uda_plugins
#     east_plugin
# )
  
# target_compile_options(${NAME} PRIVATE ${UDA_FAT_CPP_CFLAGS_OTHER} )

# include Cache
# https://github.com/catchorg/Catch2/releases/download/v2.12.3/catch.hpp
# include(FetchContent)
# FetchContent_Declare(
#   catch
#   GIT_REPOSITORY https://github.com/catchorg/Catch2.git
#   GIT_TAG        v2.12.3
# )

# # CMake 3.14+
# FetchContent_MakeAvailable(catch)

# FetchContent_GetProperties(catch)
# if(NOT catch_POPULATED)
#     FetchContent_Populate(catch)
#     add_subdirectory(${catch_SOURCE_DIR} ${catch_BINARY_DIR} EXCLUDE_FROM_ALL)    
# endif() 
