#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "atl::atl" for configuration "Release"
set_property(TARGET atl::atl APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(atl::atl PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libadios2_atl.so.2.11.0"
  IMPORTED_SONAME_RELEASE "libadios2_atl.so.2.11"
  )

list(APPEND _cmake_import_check_targets atl::atl )
list(APPEND _cmake_import_check_files_for_atl::atl "${_IMPORT_PREFIX}/lib/libadios2_atl.so.2.11.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
