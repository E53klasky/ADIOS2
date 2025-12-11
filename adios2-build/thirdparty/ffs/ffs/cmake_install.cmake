# Install script for directory: /home/jlx/Projects/CAESAR_ALL/ADIOS2/thirdparty/ffs/ffs

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

if(CMAKE_INSTALL_COMPONENT STREQUAL "adios2_ffs-libraries" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libadios2_ffs.so.2.11.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libadios2_ffs.so.2.11"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "$ORIGIN/../lib")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/lib/libadios2_ffs.so.2.11.0"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/lib/libadios2_ffs.so.2.11"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libadios2_ffs.so.2.11.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libadios2_ffs.so.2.11"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/lib:"
           NEW_RPATH "$ORIGIN/../lib")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "adios2_ffs-development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/lib/libadios2_ffs.so")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "adios2_ffs-development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/adios2/thirdparty" TYPE FILE FILES
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/ffs/ffs/ffs-config.cmake"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/ffs/ffs/ffs-config-version.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "adios2_ffs-development" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/adios2/thirdparty/ffs-targets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/adios2/thirdparty/ffs-targets.cmake"
         "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/ffs/ffs/CMakeFiles/Export/3cfc43dc0885a7bd45eed9028670f36b/ffs-targets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/adios2/thirdparty/ffs-targets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/adios2/thirdparty/ffs-targets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/adios2/thirdparty" TYPE FILE FILES "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/ffs/ffs/CMakeFiles/Export/3cfc43dc0885a7bd45eed9028670f36b/ffs-targets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/adios2/thirdparty" TYPE FILE FILES "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/ffs/ffs/CMakeFiles/Export/3cfc43dc0885a7bd45eed9028670f36b/ffs-targets-release.cmake")
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/ffs/ffs/fm/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/ffs/ffs/cod/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/ffs/ffs/ffs/cmake_install.cmake")
endif()

