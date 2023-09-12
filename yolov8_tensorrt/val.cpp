//#include <opencv2/opencv.hpp>
//#include <cuda_runtime_api.h>
//#include <NvInfer.h>
//#include <NvOnnxParser.h>
//#include <NvInferPlugin.h>
//#include "logger.h"
//#include <fstream>
//#include <iostream>
//#include <memory>
//#include <sstream>
//#include <math.h>
//#include <numeric>
//
//using namespace std;
//
//
//inline int64_t volume(const nvinfer1::Dims& d)
//{
//    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
//}
//
//
//inline unsigned int getElementSize(nvinfer1::DataType t)
//{
//    switch (t)
//    {
//    case nvinfer1::DataType::kINT32: return 4;
//    case nvinfer1::DataType::kFLOAT: return 4;
//    case nvinfer1::DataType::kHALF: return 2;
//    case nvinfer1::DataType::kBOOL:
//    case nvinfer1::DataType::kINT8: return 1;
//    }
//    throw std::runtime_error("Invalid DataType.");
//    return 0;
//}
//
//
///**
//*
//*/
//int main() {
//    int trt_version = nvinfer1::kNV_TENSORRT_VERSION_IMPL;
//    cout << "trt_version = " << trt_version << endl;    // 8601
//
//    string image_path = "D:/1/2.jpg";
//    string model_path = "D:/TensorRT-8.6.1.6/bin/yolov8n.trt";
//
//    cv::Mat image = cv::imread(image_path);
//    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
//    int origin_height = image.size().height;
//    int origin_width = image.size().width;
//
//    /***************************** preprocess *****************************/
//    // 缩放
//    int dst_size = 640;
//    float ratio = std::min((float)dst_size / origin_height, (float)dst_size / origin_width);
//    int new_height = (int)(origin_height * ratio);
//    int new_width = (int)(origin_width * ratio);
//    cout << new_height << " " << new_width << endl;                     // 640 480
//    cv::resize(image, image, cv::Size(new_width, new_height));
//
//    // 填充为正方形
//    int padding_height = dst_size - new_height;
//    int padding_width = dst_size - new_width;
//    cout << padding_height << " " << padding_width << endl;             // 0 160
//    // 填充右下角
//    cv::copyMakeBorder(image, image, 0, padding_height, 0, padding_width, cv::BorderTypes::BORDER_CONSTANT, { 114, 114, 114 });
//    //cv::imshow("0", image);
//    //cv::waitKey(0);
//
//    // 转换为float并归一化
//    image.convertTo(image, CV_32FC3, 1.0f / 255.0f, 0);
//
//    // hwc -> nchw
//    cv::Mat blob = cv::dnn::blobFromImage(image);
//    // cv::dnn::blobFromImages(); // 一次转换多张图片
//    /***************************** preprocess *****************************/
//
//    /******************************* engine *******************************/
//    /******************** load engine ********************/
//    string cached_engine;
//    std::fstream file;
//    std::cout << "loading filename from:" << model_path << std::endl;
//    file.open(model_path, std::ios::binary | std::ios::in);
//    if (!file.is_open()) {
//        std::cout << "read file error: " << model_path << std::endl;
//        cached_engine = "";
//    }
//    while (file.peek() != EOF) {
//        std::stringstream buffer;
//        buffer << file.rdbuf();
//        cached_engine.append(buffer.str());
//    }
//    file.close();
//
//    nvinfer1::IRuntime* trtRuntime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
//    initLibNvInferPlugins(&sample::gLogger, "");
//    nvinfer1::ICudaEngine* engine = trtRuntime->deserializeCudaEngine(cached_engine.data(), cached_engine.size());
//    assert(engine != nullptr);
//    std::cout << "deserialize done" << std::endl;
//    /******************** load engine ********************/
//
//    /********************** binding **********************/
//    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
//    assert(context != nullptr);
//
//    //get buffers
//    int min_batches;
//    int max_batches;
//    int nbBindings = engine->getNbBindings();
//    assert(nbBindings == 2);
//    vector<int> bufferSize(nbBindings);
//    void* cudaBuffers[2];
//    for (int i = 0; i < nbBindings; i++) {
//        const char* name;
//        int mode;
//        nvinfer1::DataType dtype;
//        nvinfer1::Dims dims;
//
//        mode = engine->bindingIsInput(i);
//        name = engine->getBindingName(i);
//        dtype = engine->getBindingDataType(i);
//        dims = context->getBindingDimensions(i);
//
//
//        int totalSize = volume(dims) * getElementSize(dtype);
//        bufferSize[i] = totalSize;
//        cudaMalloc(&cudaBuffers[i], totalSize);
//
//        fprintf(stderr, "name: %s, mode: %d, dims: [%d, %d, %d, %d], totalSize: %d Byte\n", name, mode, dims.d[0], dims.d[1], dims.d[2], dims.d[3], totalSize);
//    }
//
//    /*********************** infer ***********************/
//    int outNums = int(bufferSize[1] / sizeof(float)); // sizeof(float) = 4 Byte = 32bit
//    float* output = new float[outNums];
//
//    ///****** sync infer ******/
//    cudaMemcpy(cudaBuffers[0], blob.ptr<float>(), bufferSize[0], cudaMemcpyHostToDevice);
//    context->executeV2(cudaBuffers);
//    cudaMemcpy(output, cudaBuffers[1], bufferSize[1], cudaMemcpyDeviceToHost);
//    ///****** sync infer ******/
//    /*********************** infer ***********************/
//    /******************************* engine *******************************/
//
//    /**************************** postprocess *****************************/
//    std::ofstream outfile("./result.txt");
//    for (int i = 0; i < outNums; i++) {
//        outfile << output[i] << std::endl;
//    }
//    outfile.close();
//
//    delete[] output; // 对于基本数据类型, delete 和 delete[] 效果相同
//
//    // 析构顺序很重要
//    context->destroy();
//    engine->destroy();
//    trtRuntime->destroy();
//
//    return 0;
//}
