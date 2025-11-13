// yzy
// yolov8进行tensorrt部署的源文件
#include "rm_detector_v8s_p2/inferencer.h"
#include <ros/ros.h>

// 命名空间
using namespace cv;
using namespace nvinfer1;

namespace rm_detector_v8
{
Inferencer::Inferencer()
{
  cudaSetDevice(kGpuId);
}

Inferencer::~Inferencer()
{
  // Release stream and buffers
  cudaStreamDestroy(stream_);
  CUDA_CHECK(cudaFree(device_buffers_[0]));
  CUDA_CHECK(cudaFree(device_buffers_[1]));
  CUDA_CHECK(cudaFree(decode_ptr_device_));
  delete[] decode_ptr_host_;
  cuda_preprocess_destroy();
  // Destroy the enginestatic
  delete context_;
  delete engine_;
  delete runtime_;
}

void Inferencer::init(std::string& engine_path, Logger& logger)
{
  logger_ = &logger;

  deserialize_engine(engine_path, &runtime_, &engine_, &context_);

  CUDA_CHECK(cudaStreamCreate(&stream_));
  cuda_preprocess_init(kMaxInputImageSize);

  auto out_dims = engine_->getBindingDimensions(1);
  model_bboxes_ = out_dims.d[0];

  prepare_buffers(engine_, &device_buffers_[0], &device_buffers_[1], &decode_ptr_host_, &decode_ptr_device_);
}

void Inferencer::prepare_buffers(ICudaEngine* engine, float** input_buffer_device, float** output_buffer_device,
                                 float** decode_ptr_host, float** decode_ptr_device)
{
  assert(engine->getNbBindings() == 2);
  // In order to bind the buffers, we need to know the names of the input and output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int inputIndex = engine->getBindingIndex(kInputTensorName);
  const int outputIndex = engine->getBindingIndex(kOutputTensorName);
  assert(inputIndex == 0);
  assert(outputIndex == 1);
  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc((void**)input_buffer_device, BatchSize_ * 3 * kInputH_ * kInputW_ * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)output_buffer_device, BatchSize_ * K_OUTPUT_SIZE * sizeof(float)));

  *decode_ptr_host = new float[1 + kMaxNumOutputBbox * bbox_element];
  CUDA_CHECK(cudaMalloc((void**)decode_ptr_device, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element)));
}

void Inferencer::detect(Mat& frame)
{
  target_objects_.clear();
  cuda_batch_preprocess(frame, device_buffers_[0], kInputW_, kInputH_, stream_);
  infer(*context_, stream_, (void**)device_buffers_, BatchSize_, decode_ptr_host_, decode_ptr_device_, model_bboxes_);
  batch_process(target_objects_, decode_ptr_host_, bbox_element, frame);
}

void Inferencer::infer(IExecutionContext& context, cudaStream_t& stream, void** buffers, int batchsize,
                       float* decode_ptr_host, float* decode_ptr_device, int model_bboxes)
{
  context.enqueue(batchsize, buffers, stream, nullptr);
  CUDA_CHECK(
      cudaMemsetAsync(decode_ptr_device, 0, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), stream));
  cuda_decode((float*)buffers[1], model_bboxes, conf_thresh_, decode_ptr_device, kMaxNumOutputBbox, stream);
  cuda_nms(decode_ptr_device, nms_thresh_, kMaxNumOutputBbox, stream);  //cuda nms
  CUDA_CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device,
                             sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void Inferencer::deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine,
                                    IExecutionContext** context)
{
  std::ifstream file(engine_name, std::ios::binary);
  if (!file.good())
  {
    std::cerr << "read " << engine_name << " error!" << std::endl;
    assert(false);
  }
  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
  char* serialized_engine = new char[size];
  assert(serialized_engine);
  file.read(serialized_engine, size);
  file.close();

  *runtime = createInferRuntime(*logger_);
  assert(*runtime);
  *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
  assert(*engine);
  *context = (*engine)->createExecutionContext();
  assert(*context);
  delete[] serialized_engine;
}

}  // namespace rm_detector_v8