cmake_minimum_required(VERSION 3.12.4)
get_directory_property(hasParent PARENT_DIRECTORY)
if(NOT hasParent)
	option(unittests "unittests" OFF)
	get_filename_component(_PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
	set_property(GLOBAL PROPERTY GLOBAL_FETCHDEPS_BASE ${CMAKE_CURRENT_SOURCE_DIR}/al2o3 )
	include(FetchContent)
	FetchContent_Declare( al2o3 GIT_REPOSITORY https://github.com/DeanoC/al2o3 GIT_TAG master )
	FetchContent_GetProperties(al2o3)
	if(NOT al2o3_POPULATED)
		FetchContent_Populate(al2o3)
		add_subdirectory(${al2o3_SOURCE_DIR} ${al2o3_BINARY_DIR})
	endif()
	INIT_AL2O3(${CMAKE_CURRENT_SOURCE_DIR})
endif ()

SET_MIN_VERSIONS()
set(ProjectName syclfire)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

set(ComputeCpp_DIR "C:/Program Files/codeplay/computecpp")

# spir64, spirv64, ptx64
set(_COMPUTECPP_BACKEND "spir64")

set(COMPUTECPP_BITCODE "${_COMPUTECPP_BACKEND}" CACHE STRING "Bitcode type to use as SYCL target in compute++")

project(${ProjectName} C CXX)
find_package(ComputeCpp REQUIRED)
add_compile_definitions( _COMPUTECPP_BACKEND COMPUTECPP_BACKEND_${_COMPUTECPP_BACKEND} )

set(Src
		main.cpp
		accel_sycl.cpp
		accel_sycl.hpp
		fire.cpp
		fire.hpp
		pdcurses/pdcurses/pad.c
		pdcurses/pdcurses/addch.c
		pdcurses/pdcurses/addchstr.c
		pdcurses/pdcurses/addstr.c
		pdcurses/pdcurses/attr.c
		pdcurses/pdcurses/beep.c
		pdcurses/pdcurses/bkgd.c
		pdcurses/pdcurses/border.c
		pdcurses/pdcurses/clear.c
		pdcurses/pdcurses/color.c
		pdcurses/pdcurses/debug.c
		pdcurses/pdcurses/delch.c
		pdcurses/pdcurses/deleteln.c
		pdcurses/pdcurses/getch.c
		pdcurses/pdcurses/getstr.c
		pdcurses/pdcurses/getyx.c
		pdcurses/pdcurses/inch.c
		pdcurses/pdcurses/inchstr.c
		pdcurses/pdcurses/initscr.c
		pdcurses/pdcurses/inopts.c
		pdcurses/pdcurses/insch.c
		pdcurses/pdcurses/insstr.c
		pdcurses/pdcurses/instr.c
		pdcurses/pdcurses/kernel.c
		pdcurses/pdcurses/keyname.c
		pdcurses/pdcurses/mouse.c
		pdcurses/pdcurses/move.c
		pdcurses/pdcurses/outopts.c
		pdcurses/pdcurses/overlay.c
		pdcurses/pdcurses/pad.c
		pdcurses/pdcurses/panel.c
		pdcurses/pdcurses/printw.c
		pdcurses/pdcurses/refresh.c
		pdcurses/pdcurses/scanw.c
		pdcurses/pdcurses/scr_dump.c
		pdcurses/pdcurses/scroll.c
		pdcurses/pdcurses/slk.c
		pdcurses/pdcurses/termattr.c
		pdcurses/pdcurses/touch.c
		pdcurses/pdcurses/util.c
		pdcurses/pdcurses/window.c
		pdcurses/wincon/pdcclip.c
		pdcurses/wincon/pdcdisp.c
		pdcurses/wincon/pdcgetsc.c
		pdcurses/wincon/pdckbd.c
		pdcurses/wincon/pdcscrn.c
		pdcurses/wincon/pdcsetsc.c
		pdcurses/wincon/pdcutil.c
		pdcurses/wincon/pdcwin.h
		)
set(Deps
		al2o3_platform
		al2o3_os
		al2o3_cmath
		)
ADD_CONSOLE_APP( ${ProjectName} "${Src}" "${Deps}")
add_sycl_to_target(TARGET ${ProjectName} SOURCES
		accel_sycl.cpp
		fire.cpp
		)
include_directories( ${ProjectName} pdcurses)