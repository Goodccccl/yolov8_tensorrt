#include"infer.h"
#include"postprocess.h"
#include <sys/timeb.h> 

int main() {
	Parameters param{};
	param.batch_size = 1;
	param.input_channels = 3;
	param.input_height = 2720;
	param.input_width = 2720;
	param.conf = 0.10;
	param.iou = 0.45;
	
	struct timeb ts1_det, ts2_det;
	time_t clockBegin_det, clockEnd_det{};

	std::string img_Path = "D:\\2\\123.bmp";
	std::string trt_Path = "D:\\TensorRT-8.6.1.6\\bin\\best.trt";
	//std::string trt_Path = "D:\\TensorRT-8.6.1.6\\bin\\yolov8n_384_640.trt";
	std::vector<YOLOV8ScaleParams> vetyolovtparams;
	std::vector<Detection> outputData = infer(param, trt_Path, img_Path, vetyolovtparams);
	std::vector<Detection> nms_result = nms(outputData, param.iou);
	// 结果绘制回原图
	cv::Mat ImgResult = draw(img_Path, vetyolovtparams, nms_result);
	cv::imwrite("D:\\2\\result.jpg", ImgResult);
	return 0;
}