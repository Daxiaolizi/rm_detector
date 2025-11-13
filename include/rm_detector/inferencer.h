/*
 * @Author: Daxiaolizi
 * @Date: 2025-11-03 20:30:23
 */
// yolov12进行tensorrt部署
#ifndef INFERENCER_H
#define INFERENCER_H

#include "Infer_yolov12/cuda_utils.h"
#include "Infer_yolov12/logging.h"

#include "Infer_yolov12/postprocess.h"
#include "Infer_yolov12/preprocess.h"

#include <iostream>
#include <chrono>
#include <cmath>

using namespace nvinfer1;

namespace rm_detector
{
    class Inferencer
    {
    public:
        Inferencer();

        ~Inferencer();

        void prepare_buffers(ICudaEngine *engine, float **input_buffer_device, float **output_buffer_device,
                             float **decode_ptr_host, float **decode_ptr_device);

        void detect(cv::Mat &frame);

        void infer(IExecutionContext &context, cudaStream_t &stream, void **buffers, int batchsize,
                   float *decode_ptr_host, float *decode_ptr_device, int model_bboxes);

        void deserialize_engine(std::string &engine_name, IRuntime **runtime, ICudaEngine **engine,
                                IExecutionContext **context);

        void init(std::string &engine_path, Logger &logger);

        IRuntime *runtime_{nullptr};
        ICudaEngine *engine_{nullptr};
        IExecutionContext *context_{nullptr};
        Logger *logger_{nullptr};

        int model_bboxes_;
        float *device_buffers_[2];
        float *decode_ptr_host_ = nullptr;
        float *decode_ptr_device_ = nullptr;

        int BatchSize_ = 1;
        int kInputH_ = 640;
        int kInputW_ = 640;

        cudaStream_t stream_;

        const static int K_OUTPUT_SIZE = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
        std::vector<
        std::vector<Detection>> target_objects_;

        float conf_thresh_;
        float nms_thresh_;
    };
} // namespace rm_detector

#endif