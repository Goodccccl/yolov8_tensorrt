#pragma once
#include"includes.h"
#include "parameters.h"


/* ������ҪԤ���ͼƬ */
bool load_images(std::string imagePath, std::vector<cv::Mat> &srcImg);


/* resizeͼƬ����Ҫ�ĳߴ� */
void resize_images(cv::Mat& mat, cv::Mat& mat_rs, int target_height, int target_width, YOLOV8ScaleParams& scale_params, int& new_w, int& new_h);

void resize_images2(cv::Mat& mat, cv::Mat& mat_rs, int target_height, int target_width, YOLOV8ScaleParams& scale_params);

/* Normalization && hwc2chw */
bool normalization(cv::Mat mat, float* data);

/* Ԥ������ */
float* preprocess(std::string image_path, int target_height, int target_width, std::vector<YOLOV8ScaleParams> &vetyolovtparams);
//float* preprocess(std::string image_path, int target_height, int target_width, int& new_w, int& new_h, std::vector<YOLOV5ScaleParams> &vetyolovtparams);

cv::Mat preprocess2(std::string image_path, int target_height, int target_width);

cv::Mat preprocess3(std::string image_path, int target_height, int target_width);

cv::Mat preprocess4(std::string image_path);