#include"infer.h"
#include"parameters.h"
#include"preprocess.h"
#include "cuda_utils.h"


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

int volume(nvinfer1::Dims dims)
{
	int nb_dims = dims.nbDims;
	int result = 1;
	for (int i = 0; i < nb_dims; i++)
	{
		result = result * dims.d[i];
	}
	return result;
}

float* infer(Parameters param, const std::string enginePath, const std::string imgPath)
{
	std::vector<unsigned char> model_data = load_engine(enginePath);
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
	nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(model_data.data(), model_data.size());
	if (engine == nullptr)
	{
		printf("Deserialize cuda engine failed!\n");
		runtime->destroy();
	}
	nvinfer1::IExecutionContext* context = engine->createExecutionContext();

	int num_bindings = engine->getNbBindings();
	void* bindings[2];
	std::vector<int>bindings_mem;	// ÄÚ´æ´óÐ¡

	for (int i = 0; i < num_bindings; i++)
	{
		const char* name;
		int mode;
		nvinfer1::DataType dtype;
		nvinfer1::Dims dims;
		int totalSize;

		name = engine->getBindingName(i);
		mode = engine->bindingIsInput(i);
		dtype = engine->getBindingDataType(i);
		dims = engine->getBindingDimensions(i);

		totalSize = volume(dims) * sizeof(float);
		bindings_mem.push_back(totalSize);
		cudaMalloc(&bindings[i], totalSize);
	}
	int outputs_num = 1;
	int new_w, new_h;
	//float* input_vec = preprocess(imgPath, param.input_height, param.input_width, new_w, new_h);
	cv::Mat src = preprocess2(imgPath, param.input_height, param.input_width);

	cudaStream_t stream;
	cudaStreamCreate(&stream);
	//cudaMemcpy(bindings[0], input_vec, bindings_mem[0], cudaMemcpyHostToDevice);
	cudaMemcpy(bindings[0], src.ptr<float>(), bindings_mem[0], cudaMemcpyHostToDevice);

	std::vector<float*> outputs;
	float* output = new float[bindings_mem[1] / sizeof(float)];
	outputs.push_back(output);
	context->enqueueV2(bindings, stream, nullptr);

	cudaMemcpy(outputs[0], bindings[1], bindings_mem[1], cudaMemcpyDeviceToHost);

	std::cout << (outputs[0][100]) << std::endl;
	
	std::ofstream abc;
	abc.open("F:\\1234.txt");
	for (int i = 0; i < (bindings_mem[1] / sizeof(float)); i++)
	{
		abc << outputs[0][i] << std::endl;
	}
 	abc.close();
	
	return 0;
}