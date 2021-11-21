CMAKE_MINIMUM_REQUIRED(VERSION 3.19)
PROJECT(ZED_OPEN3D)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
message("${CMAKE_MODULE_PATH}")
IF (WIN32)
    # To include eigen in window, set env variables
    # EIGEN3_INCLUDE_DIR = C:\Program Files (x86)\Eigen3\include after installing.
    # You might have to restart the IDE after setting env
    INCLUDE_DIRECTORIES ( "$ENV{EIGEN3_INCLUDE_DIR}" )
ELSE()
    find_package(Eigen3 REQUIRED NO_MODULE)
ENDIF()

find_package(ZED 3 REQUIRED)
include_directories(${ZED_INCLUDE_DIRS})
link_directories(${ZED_LIBRARY_DIR})

find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES})
message("cuda = ${CUDA_LIBRARIES}")

SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})

find_package(OpenCV 4.5.3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")


include_directories(./include)

add_executable(RunScene exe/RunScene.cpp src/ZedUtils.cpp src/Misc.cpp src/SceneInterpreter.cpp)
target_link_libraries(RunScene PRIVATE  ${ZED_LIBS}   ${OpenCV_LIBRARIES}  ${CUDA_LIBRARIES}  )

add_executable(RunVtkTsdf test/Tsdf.cpp src/ZedUtils.cpp src/Misc.cpp src/SceneInterpreter.cpp)
target_link_libraries(RunVtkTsdf PRIVATE  ${ZED_LIBS}   ${OpenCV_LIBRARIES}  ${CUDA_LIBRARIES}  )




