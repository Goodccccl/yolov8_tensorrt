#pragma once
#include"includes.h"

typedef struct Parameters {

	int			batch_size;
	int			input_channels;
	int			input_height;
	int			input_width;
	int			output_size;
	int			Max_objects;
	int			Num_box_element;
	float		conf;
	float		iou;
};

struct alignas(float) Detection {
	float box[4];
	float conf;
	int class_id;
};

typedef struct
{
	float r; // ratio
	int dw;
	int dh;	// 左右两边填充的量
	int new_unpad_w;
	int new_unpad_h;	// 宽和高未填充前的长度
} YOLOV8ScaleParams;