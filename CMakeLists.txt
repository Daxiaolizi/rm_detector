cmake_minimum_required(VERSION 3.10)
project(rm_detector_v8)

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
set(TensorRT_VERSION "8.4.0.6")

if (NOT EXISTS "/home/ywj/opt/TensorRT-${TensorRT_VERSION}")
    MESSAGE(WARNING "/home/ywj/opt/TensorRT-${TensorRT_VERSION} not found.")
endif ()

include_directories("/home/ywj/opt/TensorRT-${TensorRT_VERSION}/include/")

link_directories("/home/ywj/opt/TensorRT-${TensorRT_VERSION}/targets/x86_64-linux-gnu/lib/")

#file(GLOB_RECURSE SRCS /home/ywj/catkin_ws/src/rm_detector_v8s_p2/utils/*.cpp /home/ywj/catkin_ws/src/rm_detector_v8s_p2/utils/*.cu)

file(GLOB_RECURSE PLUGIN_SRCS plugin/yololayer.cu)

add_library(network_ SHARED ${PLUGIN_SRCS})

file(GLOB_RECURSE SRCS src/TensorRT_yolov8/*.cpp src/TensorRT_yolov8/*.cu)

add_library(TRTx_ SHARED ${SRCS})
#add_library(TRTx_ SHARED ${SRCS})

target_link_libraries(network_
        nvinfer
        cudart
)

target_link_libraries(TRTx_ network_)

include_directories(
        include
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
        ${NGRAPH_LIBRARIES}
        ${OpenCV_LIBS}
        ${TENSORRT_LIB}
        ${CUDA_LIB}
        ${glog_LIB}
        nvinfer
        cudart
        network_
        TRTx_
)

add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)
