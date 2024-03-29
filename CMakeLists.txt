CMAKE_MINIMUM_REQUIRED(VERSION 3.19)
PROJECT(SEE_WITH_YOU)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
message("${CMAKE_MODULE_PATH}")
set(USER_NAME "JBS")
set(BUILD_TEST True)


IF (WIN32)
    # To include eigen in window, set env variables
    # EIGEN3_INCLUDE_DIR = C:\Program Files (x86)\Eigen3\include after installing.
    # You might have to restart the IDE after setting env
    INCLUDE_DIRECTORIES ( "$ENV{EIGEN3_INCLUDE_DIR}" )
ELSE()
    find_package(Eigen3 REQUIRED NO_MODULE)
ENDIF()

# GLM
find_package(glm REQUIRED)

# ZED
find_package(ZED 3.7 REQUIRED)
include_directories(${ZED_INCLUDE_DIRS})
link_directories(${ZED_LIBRARY_DIR})

# CUDA
find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES})
message("cuda = ${CUDA_LIBRARIES}")

SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})

# OPENCV
find_package(OpenCV 4.5 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")

# OPEN3D
find_package(Open3D REQUIRED)
# The options need to be the same as Open3D's default
# If Open3D is configured and built with custom options, you'll also need to
# specify the same custom options.
option(STATIC_WINDOWS_RUNTIME "Use static (MT/MTd) Windows runtime" ON)
if(STATIC_WINDOWS_RUNTIME)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()


# FOR ARGUMENT PARSING
find_package(yaml-cpp REQUIRED)


add_subdirectory(googletest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})


# SEE_WITH_YOU
###############################################################################
include_directories(./include)
add_library(ZedWrapper  src/ZedUtils.cpp src/Misc.cpp ) # https://jhnyang.tistory.com/42
target_link_libraries(ZedWrapper ${ZED_LIBS} ${Open3D_LIBRARIES} ${OpenCV_LIBRARIES} yaml-cpp)

add_library(SceneInterpreter src/SceneInterpreter.cpp src/RenderUtils.cpp src/Open3dUtils.cpp)
target_link_libraries(SceneInterpreter ZedWrapper glm::glm)
# UNIT TESTING
###############################################################################

if(BUILD_TEST)
add_executable(ZedTest test/Zed.cpp)
target_link_libraries(ZedTest  gtest_main ZedWrapper)

add_executable(SceneTest test/Scene.cpp)
target_link_libraries(SceneTest  gtest_main SceneInterpreter)
endif()

if(WIN32)
    get_target_property(open3d_type Open3D::Open3D TYPE)
    if(open3d_type STREQUAL "SHARED_LIBRARY")
        message(STATUS "Copying Open3D.dll to ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
        add_custom_command(TARGET RunOpen3DTsdf POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_INSTALL_PREFIX}/bin/Open3D.dll
                ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>)
    endif()
endif()




