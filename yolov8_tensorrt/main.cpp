#include"infer.h"

int main() {
	Parameters param{};
	param.batch_size = 1;
	param.input_channels = 3;
	param.input_height = 640;
	param.input_width = 640;
	param.Max_objects = 1024;
	param.Num_box_element = 6;
	

	std::string img_Path = "D:\\1\\2.jpg";
	std::string trt_Path = "D:\\TensorRT-8.6.1.6\\bin\\yolov8n.trt";
	//std::string trt_Path = "D:\\TensorRT-8.6.1.6\\bin\\yolov8n_384_640.trt";

	float* outputData = infer(param, trt_Path, img_Path);
	std::cout << outputData << std::endl;
}