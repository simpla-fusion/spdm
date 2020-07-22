# find_package( Boost REQUIRED )
# find_package( HDF5 REQUIRED )

include(CTest)
enable_testing()



include_directories(../source)
include_directories(../extern)


set( TESTS
   test_iterator
  #  test_document
  #  test_entry
  #  test_xml
  #  test_imas_east_magnetics
  #  test_imas_east_equilibrium
  # # test_imas_jet_tf
  # test_imas_east_pf_active
  #  test_imas_east_wall
  # test_imas_jet_iron_core
  # test_imas_jet_mse
  # test_imas_jet_thomson_scattering
  # test_imas_tcv_magnetics
  # test_imas_tcv_pf_active
  # test_imas_aug_magnetics
  # test_imas_aug_pf_active
)

foreach( TEST ${TESTS} )
  add_executable(${TEST} ${TEST}.cpp )
  # target_link_libraries(${TEST} spdb)
  add_test( ${TEST}  ${TEST} )
endforeach()
