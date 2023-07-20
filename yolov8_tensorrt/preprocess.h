#pragma once
#include"includes.h"

typedef struct
{
	float r; // ratio
	int dw;
	int dh;	// 左右两边填充的量
	int new_unpad_w;
	int new_unpad_h;	// 宽和高未填充前的长度
} YOLOV5ScaleParams;


/* 加载需要预测的图片 */
bool load_images(std::string imagePath, std::vector<cv::Mat> &srcImg);


/* resize图片到需要的尺寸 */
void resize_images(cv::Mat& mat, cv::Mat& mat_rs, int target_height, int target_width, YOLOV5ScaleParams& scale_params);

/* Normalization && hwc2chw */
bool normalization(cv::Mat mat, float* data);

/* 预处理总 */
float* preprocess(std::string image_path, int target_height, int target_width);