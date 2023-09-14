#include"infer.h"
#include"postprocess.h"

int main() {
	Parameters param{};
	param.batch_size = 1;
	param.input_channels = 3;
	param.input_height = 640;
	param.input_width = 640;
	param.conf = 0.15;
	param.iou = 0.45;
	

	std::string img_Path = "D:\\1\\2.jpg";
	std::string trt_Path = "D:\\TensorRT-8.6.1.6\\bin\\yolov8n.trt";
	//std::string trt_Path = "D:\\TensorRT-8.6.1.6\\bin\\yolov8n_384_640.trt";
	std::vector<YOLOV5ScaleParams> vetyolovtparams;
	std::vector<Detection> outputData = infer(param, trt_Path, img_Path, vetyolovtparams);
	std::vector<Detection> nms_result = nms(outputData, param.iou);
	cv::Mat ImgResult = draw(img_Path, vetyolovtparams, nms_result);
	cv::imwrite("D:\\1\\cpp\\3.jpg", ImgResult);
	return 0;
}