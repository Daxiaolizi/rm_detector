cmake_minimum_required(VERSION 3.10)
project(rm_detector)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
endif (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# CUDA
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        dynamic_reconfigure
        std_msgs
        cv_bridge
        nodelet
        pluginlib
        rm_msgs
)
#find_package(OpenCV REQUIRED)
#find_package(OpenCV 4 REQUIRED)
find_package(OpenCV 4 REQUIRED)

include_directories(/usr/local/cuda/include)

link_directories(/usr/local/cuda/lib64)

# TensorRT
set(TensorRT_VERSION "8.5.1.7")

if (NOT EXISTS "/usr/local/TensorRT-${TensorRT_VERSION}")
    MESSAGE(WARNING "/usr/local/TensorRT-${TensorRT_VERSION} not found.")
endif ()

include_directories("/usr/local/TensorRT-${TensorRT_VERSION}/include/")

link_directories("/usr/local/TensorRT-${TensorRT_VERSION}/targets/x86_64-linux-gnu/lib/")

file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/utils/*.cpp ${PROJECT_SOURCE_DIR}/utils/*.cu)

file(GLOB_RECURSE PLUGIN_SRCS ./plugin/*.cu)

add_library(network SHARED ${PLUGIN_SRCS})

add_library(TRTx SHARED ${SRCS})

target_link_libraries(network
        nvinfer
        cudart
)

target_link_libraries(TRTx network)

include_directories(
        include
        ./utils/
        ./plugin/
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

link_directories(${OpenCV_LIBRARY_DIRS})

generate_dynamic_reconfigure_options(
        cfg/dynamic.cfg
)

catkin_package(
        INCLUDE_DIRS
        include
        LIBRARIES
        CATKIN_DEPENDS
        roscpp
        nodelet
        dynamic_reconfigure
        std_msgs
        cv_bridge
        rm_msgs
        LIBRARIES ${PROJECT_NAME}
)

if (NOT EXISTS "/usr/local/cuda/include")
    MESSAGE(WARNING "/usr/local/cuda/include not found.")
endif ()

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        "/usr/local/cuda/include"
        /usr/include/eigen3
)

add_library(${PROJECT_NAME}
        src/detector.cpp
        src/inferencer.cpp
        src/rm_bytetrack/BYTETracker.cpp
        src/rm_bytetrack/kalmanFilter.cpp
        src/rm_bytetrack/lapjv.cpp
        src/rm_bytetrack/STrack.cpp
)

target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${TENSORRT_LIB}
        ${CUDA_LIB}
        nvinfer
        cudart
        network
        TRTx
)

add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)