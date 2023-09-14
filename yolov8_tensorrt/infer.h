#pragma once
#include"includes.h"
#include"parameters.h"

/* ¼ÓÔØÄ£ĞÍ */
std::vector<unsigned char> load_engine(const std::string enginePath);

std::vector<Detection> infer(Parameters param, const std::string enginePath, const std::string imgPath, std::vector<YOLOV5ScaleParams> &vetyolovtparams);
