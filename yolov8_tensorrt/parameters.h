#pragma once
#include"includes.h"

typedef struct Parameters {

	int			batch_size;
	int			input_channels;
	int					input_height;
	int					input_width;
	int			output_size;
	int			Max_objects;
	int			Num_box_element;
};