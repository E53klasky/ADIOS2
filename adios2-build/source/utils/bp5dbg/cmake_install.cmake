# Install script for directory: /home/jlx/Projects/CAESAR_ALL/ADIOS2/source/utils/bp5dbg

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

if(CMAKE_INSTALL_COMPONENT STREQUAL "adios2_scripts-runtime" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE PROGRAM RENAME "bp5dbg" FILES "/home/jlx/Projects/CAESAR_ALL/ADIOS2/source/utils/bp5dbg/bp5dbg.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "adios2_scripts-runtime" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/local/lib/python3.12/dist-packages/adios2/bp5dbg" TYPE FILE FILES
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/source/utils/bp5dbg/adios2/bp5dbg/__init__.py"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/source/utils/bp5dbg/adios2/bp5dbg/utils.py"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/source/utils/bp5dbg/adios2/bp5dbg/metadata.py"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/source/utils/bp5dbg/adios2/bp5dbg/metametadata.py"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/source/utils/bp5dbg/adios2/bp5dbg/idxtable.py"
    )
endif()

