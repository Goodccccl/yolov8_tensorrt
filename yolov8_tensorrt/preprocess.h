#pragma once
#include"includes.h"

typedef struct
{
	float r; // ratio
	int dw;
	int dh;	// ��������������
	int new_unpad_w;
	int new_unpad_h;	// ��͸�δ���ǰ�ĳ���
} YOLOV5ScaleParams;


/* ������ҪԤ���ͼƬ */
bool load_images(std::string imagePath, std::vector<cv::Mat> &srcImg);


/* resizeͼƬ����Ҫ�ĳߴ� */
void resize_images(cv::Mat& mat, cv::Mat& mat_rs, int target_height, int target_width, YOLOV5ScaleParams& scale_params);

/* Normalization && hwc2chw */
bool normalization(cv::Mat mat, float* data);

/* Ԥ������ */
float* preprocess(std::string image_path, int target_height, int target_width);