// yzy
// yolov5进行tensorrt部署的源文件
#include "rm_detector/inferencer.h"
#include <ros/ros.h>

// 命名空间
using namespace cv;
using namespace nvinfer1;

namespace rm_detector
{
Inferencer::Inferencer()
{
  cudaSetDevice(kGpuId);
}

Inferencer::~Inferencer()
{
  // Release stream and buffers
  //        cudaStreamDestroy(stream_);
  CUDA_CHECK(cudaFree(gpu_buffers_[0]));
  CUDA_CHECK(cudaFree(gpu_buffers_[1]));
  delete[] cpu_output_buffer_;
  cudaProprecess_.cuda_preprocess_destroy();
  // Destroy the engine
  context_->destroy();
  engine_->destroy();
  runtime_->destroy();
}

void Inferencer::init(std::string& engine_path, Logger& logger)
{
  logger_ = &logger;

  deserialize_engine(engine_path, &runtime_, &engine_, &context_);

  cudaProprecess_.cuda_preprocess_init(kMaxInputImageSize);

  prepare_buffers(engine_, &gpu_buffers_[0], &gpu_buffers_[1], &cpu_output_buffer_);
}

void Inferencer::prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer,
                                 float** cpu_output_buffer)
{
  assert(engine->getNbBindings() == 2);
  // In order to bind the buffers, we need to know the names of the input and output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int inputIndex = engine->getBindingIndex(kInputTensorName);
  const int outputIndex = engine->getBindingIndex(kOutputTensorName);
  assert(inputIndex == 0);
  assert(outputIndex == 1);
  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, kBatchSize * K_OUTPUT_SIZE * sizeof(float)));

  *cpu_output_buffer = new float[kBatchSize * K_OUTPUT_SIZE];
}

void Inferencer::detect(Mat& frame)
{
  target_objects_.clear();
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaProprecess_.cuda_batch_preprocess(frame, gpu_buffers_[0], kInputW, kInputH, stream);

  infer(*context_, stream, (void**)gpu_buffers_, cpu_output_buffer_, kBatchSize);

  batch_nms(target_objects_, cpu_output_buffer_, conf_thresh_, nms_thresh_);

  cudaStreamDestroy(stream);

}

void Inferencer::infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output,
                       int batchsize)
{
  context.enqueue(batchsize, gpu_buffers, stream, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1], batchsize * K_OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost,
                             stream));
  cudaStreamSynchronize(stream);
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

}  // namespace rm_detector