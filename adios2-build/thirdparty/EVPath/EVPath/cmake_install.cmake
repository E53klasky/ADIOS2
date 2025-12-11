# Install script for directory: /home/jlx/Projects/CAESAR_ALL/ADIOS2/thirdparty/EVPath/EVPath

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

if(CMAKE_INSTALL_COMPONENT STREQUAL "cmselect" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmselect.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmselect.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmselect.so"
         RPATH "$ORIGIN/../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11" TYPE MODULE FILES "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/EVPath/EVPath/lib/adios2-evpath-modules-2_11/libadios2_cmselect.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmselect.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmselect.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmselect.so"
         OLD_RPATH "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/lib:"
         NEW_RPATH "$ORIGIN/../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmselect.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cmselect" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/EVPath/EVPath/CMakeFiles/cmselect.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cmsockets" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmsockets.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmsockets.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmsockets.so"
         RPATH "$ORIGIN/../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11" TYPE MODULE FILES "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/EVPath/EVPath/lib/adios2-evpath-modules-2_11/libadios2_cmsockets.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmsockets.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmsockets.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmsockets.so"
         OLD_RPATH "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/lib:"
         NEW_RPATH "$ORIGIN/../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmsockets.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cmsockets" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/EVPath/EVPath/CMakeFiles/cmsockets.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cmudp" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmudp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmudp.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmudp.so"
         RPATH "$ORIGIN/../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11" TYPE MODULE FILES "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/EVPath/EVPath/lib/adios2-evpath-modules-2_11/libadios2_cmudp.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmudp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmudp.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmudp.so"
         OLD_RPATH "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/lib:"
         NEW_RPATH "$ORIGIN/../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmudp.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cmudp" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/EVPath/EVPath/CMakeFiles/cmudp.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cmmulticast" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmmulticast.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmmulticast.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmmulticast.so"
         RPATH "$ORIGIN/../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11" TYPE MODULE FILES "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/EVPath/EVPath/lib/adios2-evpath-modules-2_11/libadios2_cmmulticast.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmmulticast.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmmulticast.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmmulticast.so"
         OLD_RPATH "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/lib:"
         NEW_RPATH "$ORIGIN/../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmmulticast.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cmmulticast" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/EVPath/EVPath/CMakeFiles/cmmulticast.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cmepoll" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmepoll.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmepoll.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmepoll.so"
         RPATH "$ORIGIN/../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11" TYPE MODULE FILES "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/EVPath/EVPath/lib/adios2-evpath-modules-2_11/libadios2_cmepoll.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmepoll.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmepoll.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmepoll.so"
         OLD_RPATH "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/lib:"
         NEW_RPATH "$ORIGIN/../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmepoll.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cmepoll" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/EVPath/EVPath/CMakeFiles/cmepoll.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cmenet" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmenet.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmenet.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmenet.so"
         RPATH "$ORIGIN/../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11" TYPE MODULE FILES "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/EVPath/EVPath/lib/adios2-evpath-modules-2_11/libadios2_cmenet.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmenet.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmenet.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmenet.so"
         OLD_RPATH "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/lib:"
         NEW_RPATH "$ORIGIN/../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/adios2-evpath-modules-2_11/libadios2_cmenet.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cmenet" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/EVPath/EVPath/CMakeFiles/cmenet.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "adios2_evpath-development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/adios2/thirdparty" TYPE DIRECTORY FILES "/home/jlx/Projects/CAESAR_ALL/ADIOS2/thirdparty/EVPath/EVPath/cmake/" FILES_MATCHING REGEX "/Find[^/]*\\.cmake$" REGEX "/CMake[^/]*\\.cmake$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "adios2_evpath-libraries" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libadios2_evpath.so.2.11.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libadios2_evpath.so.2.11"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "$ORIGIN/../lib")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/lib/libadios2_evpath.so.2.11.0"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/lib/libadios2_evpath.so.2.11"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libadios2_evpath.so.2.11.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libadios2_evpath.so.2.11"
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

if(CMAKE_INSTALL_COMPONENT STREQUAL "adios2_evpath-development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/lib/libadios2_evpath.so")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "adios2_evpath-development" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/adios2/thirdparty/EVPathTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/adios2/thirdparty/EVPathTargets.cmake"
         "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/EVPath/EVPath/CMakeFiles/Export/3cfc43dc0885a7bd45eed9028670f36b/EVPathTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/adios2/thirdparty/EVPathTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/adios2/thirdparty/EVPathTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/adios2/thirdparty" TYPE FILE FILES "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/EVPath/EVPath/CMakeFiles/Export/3cfc43dc0885a7bd45eed9028670f36b/EVPathTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/adios2/thirdparty" TYPE FILE FILES "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/EVPath/EVPath/CMakeFiles/Export/3cfc43dc0885a7bd45eed9028670f36b/EVPathTargets-release.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "adios2_evpath-development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/adios2/thirdparty" TYPE FILE FILES
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/EVPath/EVPath/EVPathConfigCommon.cmake"
    "/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/thirdparty/EVPath/EVPath/EVPathConfigVersion.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "adios2_evpath-development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/adios2/thirdparty" TYPE FILE RENAME "EVPathConfig.cmake" FILES "/home/jlx/Projects/CAESAR_ALL/ADIOS2/thirdparty/EVPath/EVPath/EVPathConfigInstall.cmake")
endif()

