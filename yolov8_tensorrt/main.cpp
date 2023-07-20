#include"infer.h"

int main() {
	Parameters param{};
	param.batch_size = 1;
	param.input_channels = 3;
	param.input_height = 64;
	param.input_width = 64;
	param.Max_objects = 1024;
	param.Num_box_element = 7;
	param.output_size = 2 * param.input_height * param.input_width;

	std::string img_Path = "F:\\test\\12.jpg";
	std::string trt_Path = "F:/Artificial_neural_Network/yolov8-main/weights/onnx/yolov8n.trt";

	infer(param, trt_Path, img_Path);
}