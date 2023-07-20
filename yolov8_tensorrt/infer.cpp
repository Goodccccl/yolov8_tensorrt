#include"infer.h"
#include"parameters.h"
#include"preprocess.h"

std::vector<unsigned char> load_engine(const std::string enginePath)
{
	std::ifstream in(enginePath, std::ios::in | std::ios::binary);
	if (!in.is_open())
	{
		return {};
	}
	in.seekg(0, std::ios::end);
	size_t length = in.tellg();

	std::vector<uint8_t> data;
	if (length > 0)
	{
		in.seekg(0, std::ios::beg);
		data.resize(length);
		in.read((char*)&data[0], length);
	}
	in.close();
	return data;
}

cv::Mat* load_image(const std::string imgPath)
{
	cv::Mat ori_img = cv::imread(imgPath, cv::IMREAD_COLOR);
	cv::Mat ori_img1;
	cv::cvtColor(ori_img, ori_img1, cv::COLOR_BGR2RGB);
	cv::Mat* ori_img_ptr = new cv::Mat(ori_img1);
	return ori_img_ptr;
}

std::vector<unsigned char> infer(Parameters param, const std::string enginePath, const std::string imgPath)
{
	std::vector<unsigned char> model_data = load_engine(enginePath);
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
	nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(model_data.data(), model_data.size());
	if (engine == nullptr) {
		printf("Deserialize cuda engine failed.\n");
		runtime->destroy();
	}
	nvinfer1::IExecutionContext* context = engine->createExecutionContext();	// 上下文
	cudaStream_t stream = nullptr;
	// 创建cuda流
	cudaStreamCreate(&stream);

	// cuda malloc
	void* input_mem{ nullptr };
	cudaMalloc(&input_mem, param.batch_size * param.input_channels * param.input_height * param.input_width * sizeof(float));
	void* output_mem{ nullptr };
	cudaMalloc(&output_mem, param.batch_size * param.output_size * sizeof(float));
	float* inputData_vec = preprocess(imgPath, param.input_height, param.input_width);
	cudaMemcpyAsync(input_mem, inputData_vec, param.batch_size * sizeof(*inputData_vec) / sizeof(float), cudaMemcpyHostToDevice, stream);
	// 执行推理
	void* bindings[] = { input_mem, output_mem };
	auto start = std::chrono::system_clock::now();
	context->enqueueV2(bindings, stream, nullptr);
	auto end = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
	//拷贝结果
	std::vector<float>outputData;
	cudaMemcpyAsync(outputData.data(), output_mem, sizeof(float) * (1 + param.Max_objects * param.Num_box_element), cudaMemcpyDeviceToHost, stream);
	std::cout << outputData.data() << std::endl;
	std::vector<unsigned char> a;
	return a;
}