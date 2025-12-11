# Install script for directory: /home/jlx/Projects/CAESAR_ALL/ADIOS2/source/adios2/toolkit/sst/util

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

if(CMAKE_INSTALL_COMPONENT STREQUAL "adios2_tools-runtime" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/sst_conn_tool" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/sst_conn_tool")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/sst_conn_tool"
         RPATH "$ORIGIN/../lib:/home/jlx/Projects/CAESAR_ALL/CAESAR_C/install/lib:/home/jlx/Software/miniforge3/envs/py311torch/lib/python3.11/site-packages/torch/lib:/home/jlx/Software/nvcomp/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openmpi/lib:/home/jlx/Software/miniforge3/envs/py311torch/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/bin/sst_conn_tool")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/sst_conn_tool" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/sst_conn_tool")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/sst_conn_tool"
         OLD_RPATH "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/lib:/home/jlx/Projects/CAESAR_ALL/CAESAR_C/install/lib:/home/jlx/Software/miniforge3/envs/py311torch/lib/python3.11/site-packages/torch/lib:/home/jlx/Software/nvcomp/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openmpi/lib:/home/jlx/Software/miniforge3/envs/py311torch/lib:"
         NEW_RPATH "$ORIGIN/../lib:/home/jlx/Projects/CAESAR_ALL/CAESAR_C/install/lib:/home/jlx/Software/miniforge3/envs/py311torch/lib/python3.11/site-packages/torch/lib:/home/jlx/Software/nvcomp/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openmpi/lib:/home/jlx/Software/miniforge3/envs/py311torch/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/sst_conn_tool")
    endif()
  endif()
endif()

