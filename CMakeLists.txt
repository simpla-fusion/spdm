cmake_minimum_required( VERSION 3.15)

cmake_policy( SET CMP0042 NEW )
cmake_policy( SET CMP0048 NEW )
cmake_policy( SET CMP0068 NEW )
cmake_policy( SET CMP0074 OLD )


project(SpDB)

# Extra cmake modules to find libraries
set( CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  ${CMAKE_SOURCE_DIR}/cmake
  ${CMAKE_SOURCE_DIR}/cmake/Modules
)

set (CMAKE_CXX_STANDARD 20)

add_subdirectory( source )
add_subdirectory( tests )
add_subdirectory( extern )
