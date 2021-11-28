CMAKE_MINIMUM_REQUIRED(VERSION 3.19)
PROJECT(SEE_WITH_YOU)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
message("${CMAKE_MODULE_PATH}")
IF (WIN32)
    # To include eigen in window, set env variables
    # EIGEN3_INCLUDE_DIR = C:\Program Files (x86)\Eigen3\include after installing.
    # You might have to restart the IDE after setting env
    INCLUDE_DIRECTORIES ( "$ENV{EIGEN3_INCLUDE_DIR}" )
    # TBB library add. Still IDK why should I manually do this linking on my own.
    # TBB_DIR = <path/to/open3d>\Open3D\build\tbb\src\ext_tbb-build\Release
    link_directories("$ENV{TBB_DIR}")
ELSE()
    find_package(Eigen3 REQUIRED NO_MODULE)
ENDIF()

message($ENV{TBB_DIR})

# OpenGL

# ZED
find_package(ZED 3 REQUIRED)
include_directories(${ZED_INCLUDE_DIRS})
link_directories(${ZED_LIBRARY_DIR})

# CUDA
find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES})
message("cuda = ${CUDA_LIBRARIES}")

SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})

# OPENCV
find_package(OpenCV 4.5.3 REQUIRED)
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


# SEE_WITH_YOU
include_directories(./include)

add_executable(RunScene exe/RunScene.cpp src/ZedUtils.cpp src/Misc.cpp src/SceneInterpreter.cpp src/Open3dUtils.cpp)
target_link_libraries(RunScene PRIVATE  ${ZED_LIBS}   ${OpenCV_LIBRARIES}  ${CUDA_LIBRARIES} Open3D::Open3D )

add_executable(RunVtkTsdf test/TsdfOpenCV.cpp src/ZedUtils.cpp src/Misc.cpp src/SceneInterpreter.cpp )
target_link_libraries(RunVtkTsdf PRIVATE  ${ZED_LIBS}   ${OpenCV_LIBRARIES}  ${CUDA_LIBRARIES} Open3D::Open3D )

add_executable(RunOpen3dOpengl test/TsdfOpen3dOpengl.cpp src/ZedUtils.cpp src/Misc.cpp src/SceneInterpreter.cpp src/Open3dUtils.cpp)
target_link_libraries(RunOpen3dOpengl PRIVATE  ${ZED_LIBS}   ${OpenCV_LIBRARIES}  ${CUDA_LIBRARIES} Open3D::Open3D tbb_static )

add_executable(RunOpen3DTsdf test/TsdfOpen3d.cpp src/ZedUtils.cpp src/Misc.cpp src/SceneInterpreter.cpp src/Open3dUtils.cpp)
target_link_libraries(RunOpen3DTsdf PRIVATE  ${ZED_LIBS}   ${OpenCV_LIBRARIES}  ${CUDA_LIBRARIES} Open3D::Open3D tbb_static )


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



