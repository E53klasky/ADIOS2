set(_ADIOS2_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
list(INSERT CMAKE_MODULE_PATH 0 "/home/jlx/Projects/CAESAR_ALL/ADIOS2/cmake")

if(NOT ON)
  set(atl_DIR /home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/atl/atl)
  set(dill_DIR /home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/dill/dill)
  set(ffs_DIR /home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/ffs/ffs)
endif()

if(TRUE)
  set(EVPath_DIR /home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/EVPath/EVPath)
  if(NOT ON)
    set(enet_DIR /home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/enet/enet)
  endif()
endif()

set(${CMAKE_FIND_PACKAGE_NAME}_CONFIG "${CMAKE_CURRENT_LIST_FILE}")
include("${CMAKE_CURRENT_LIST_DIR}/adios2-config-common.cmake")

set(CMAKE_MODULE_PATH ${_ADIOS2_CMAKE_MODULE_PATH})
unset(_ADIOS2_CMAKE_MODULE_PATH)
