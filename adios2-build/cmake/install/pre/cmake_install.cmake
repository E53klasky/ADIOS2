# Install script for directory: /home/jlx/Projects/CAESAR_ALL/ADIOS2/cmake/install/pre

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/jlx/Projects/CAESAR_ALL/ADIOS2/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "adios2_core-development" OR NOT CMAKE_INSTALL_COMPONENT)
  
  message("Pre-installation cleanup of CMake files")
  file(REMOVE_RECURSE "/home/jlx/Projects/CAESAR_ALL/ADIOS2/install/lib/cmake/adios2")

endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "adios2_core-development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/adios2" TYPE FILE FILES
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/cmake/CMakeFindDependencyMacro.cmake"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/cmake/FindBZip2.cmake"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/cmake/FindCaliper.cmake"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/cmake/FindCrayDRC.cmake"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/cmake/FindDAOS.cmake"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/cmake/FindDataSpaces.cmake"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/cmake/FindHDF5.cmake"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/cmake/FindIME.cmake"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/cmake/FindLIBFABRIC.cmake"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/cmake/FindMPI.cmake"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/cmake/FindPkgConfig.cmake"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/cmake/FindPython.cmake"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/cmake/FindPythonModule.cmake"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/cmake/FindSZ.cmake"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/cmake/FindSodium.cmake"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/cmake/FindUCX.cmake"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/cmake/FindZeroMQ.cmake"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/cmake/Findpugixml.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "adios2_core-development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/adios2/upstream" TYPE DIRECTORY FILES "/home/jlx/Projects/CAESAR_ALL/ADIOS2/cmake/upstream/")
endif()

