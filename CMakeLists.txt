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
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)
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

find_package(OpenCV 4 REQUIRED)

include_directories(/usr/local/cuda-11.8/include)

link_directories(/usr/local/cuda-11.8/lib64)

#TensorRT
set(TensorRT_VERSION "8.5.1.7")

if (NOT EXISTS "/usr/local/TensorRT-${TensorRT_VERSION}")
    MESSAGE(WARNING "/usr/local/TensorRT-${TensorRT_VERSION} not found.")
endif ()

include_directories("/usr/local/TensorRT-${TensorRT_VERSION}/include/")
link_directories("/usr/local/TensorRT-${TensorRT_VERSION}/targets/x86_64-linux-gnu/lib/")

file(GLOB_RECURSE PLUGIN_SRCS plugin/yololayer.cu)

add_library(network_ SHARED ${PLUGIN_SRCS})

file(GLOB_RECURSE SRCS src/Infer_yolov12/*.cpp src/Infer_yolov12/*.cu)

add_library(TRTx_ SHARED ${SRCS})

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
 INCLUDE_DIRS include/

 LIBRARIES  ${PROJECT_NAME}
 CATKIN_DEPENDS cv_bridge dynamic_reconfigure nodelet rm_msgs roscpp rospy std_msgs
 
 DEPENDS system_lib
)

if (NOT EXISTS "/usr/local/cuda-11.8/include")
    MESSAGE(WARNING "/usr/local/cuda-11.8/include not found.")
endif ()

include_directories(
include
  ${catkin_INCLUDE_DIRS}
  "/usr/local/cuda-11.8/include"
  /usr/include/eigen3
)

add_library(${PROJECT_NAME} 
src/rm_detector/detector.cpp
src/rm_detector/inferencer.cpp
src/Infer_yolov12/block.cpp
src/Infer_yolov12/model.cpp
src/Infer_yolov12/postprocess.cpp

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