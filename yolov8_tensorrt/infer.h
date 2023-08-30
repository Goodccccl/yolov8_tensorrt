#pragma once
#include"includes.h"
#include"parameters.h"

/* ¼ÓÔØÄ£ĞÍ */
std::vector<unsigned char> load_engine(const std::string enginePath);

float* infer(Parameters param, const std::string enginePath, const std::string imgPath);
