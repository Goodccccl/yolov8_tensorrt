#pragma once
#include"parameters.h"
#include<opencv.hpp>

std::vector<Detection> nms(std::vector<Detection> outputs_arrange, float threshold);

cv::Mat draw(std::string src_imgPath, std::vector<YOLOV8ScaleParams> vetyolovtparams, std::vector<Detection> nms_result);