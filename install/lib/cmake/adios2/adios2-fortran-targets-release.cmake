#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "adios2::fortran" for configuration "Release"
set_property(TARGET adios2::fortran APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(adios2::fortran PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "adios2::c;adios2::core"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libadios2_fortran.so.2.11.0"
  IMPORTED_SONAME_RELEASE "libadios2_fortran.so.2.11"
  )

list(APPEND _cmake_import_check_targets adios2::fortran )
list(APPEND _cmake_import_check_files_for_adios2::fortran "${_IMPORT_PREFIX}/lib/libadios2_fortran.so.2.11.0" )

# Import target "adios2::fortran_mpi" for configuration "Release"
set_property(TARGET adios2::fortran_mpi APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(adios2::fortran_mpi PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "adios2::c_mpi"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libadios2_fortran_mpi.so.2.11.0"
  IMPORTED_SONAME_RELEASE "libadios2_fortran_mpi.so.2.11"
  )

list(APPEND _cmake_import_check_targets adios2::fortran_mpi )
list(APPEND _cmake_import_check_files_for_adios2::fortran_mpi "${_IMPORT_PREFIX}/lib/libadios2_fortran_mpi.so.2.11.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
