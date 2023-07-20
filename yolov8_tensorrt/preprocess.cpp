#include"preprocess.h"


bool load_images(std::string imagePath, std::vector<cv::Mat> &srcImg)
{
	if (imagePath.size() == 0)
	{
		return false;
	}
	else
	{
		cv::Mat srcimg = cv::imread(imagePath, 1);
		if (srcimg.channels() == 1)
		{
			cv::cvtColor(srcimg, srcimg, cv::COLOR_GRAY2BGR);
			srcImg.push_back(srcimg);
		}
		else if (srcimg.channels() == 4)
		{
			cv::cvtColor(srcimg, srcimg, cv::COLOR_BGRA2BGR);
			srcImg.push_back(srcimg);
		}
		else
		{
			srcImg.push_back(srcimg);
		}
		return true;
	}
}


void resize_images(cv::Mat& mat, cv::Mat& mat_rs, int target_height, int target_width, YOLOV5ScaleParams& scale_params)
{
	if (mat.empty()) return;
	int img_height = static_cast<int>(mat.rows);
	int img_width = static_cast<int>(mat.cols);
	// 创建输出图像
	mat_rs = cv::Mat(target_height, target_width, CV_8UC3, cv::Scalar(114, 114, 144));
	//cv::imshow("mat_rs", mat_rs);
	//cv::waitKey(1000);
	// 获取缩放ratio
	float w_r = float(target_width) / float(img_width);
	float h_r = float(target_height) / float(img_height);
	float r = std::min(w_r, h_r);
	// 得到未填充的缩放尺寸
	int new_unpad_w = static_cast<int>((float)img_width * r);
	int new_unpad_h = static_cast<int>((float)img_height * r);
	int pad_w = target_width - new_unpad_w;
	int pad_h = target_height - new_unpad_h;

	int dw = pad_w / 2;
	int dh = pad_h / 2;

	// resize
	cv::Mat new_unpad_mat = mat.clone();
	//cv::imshow("new_unpad_mat1", new_unpad_mat);
	//cv::waitKey(1000);
	cv::resize(new_unpad_mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
	//cv::imshow("new_unpad_mat2", new_unpad_mat);
	//cv::waitKey(1000);
	new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h))); // 复制到输出图像上
	//cv::imshow("new_mat_rs",mat_rs);
	//cv::waitKey(1000);

	// 记录resize参数
	scale_params.r = r;
	scale_params.dw = dw;
	scale_params.dh = dh;
	scale_params.new_unpad_w = new_unpad_w;
	scale_params.new_unpad_h = new_unpad_h;
}

/* 归一化 && hwc2chw && bgr2rgb */
bool normalization(cv::Mat mat, float *data)
{
	if (mat.empty()) return false;
	int i = 0;
	//std::cout << mat.rows << " and " << mat.cols << std::endl;
	for (int row = 0; row < mat.rows; ++row) {
		uchar* uc_pixel = mat.data + row * mat.step;
		//std::cout << "row="<< row << std::endl;
		for (int col = 0; col < mat.cols; ++col) {
			data[i] = (float)uc_pixel[2] / 255.0;
			//std::cout << (float)uc_pixel[2] << std::endl;
			//std::cout << data[i] << std::endl;
			data[i + mat.rows * mat.cols] = (float)uc_pixel[1] / 255.0;
			//std::cout << (float)uc_pixel[1] << std::endl;
			//std::cout << data[i + mat.rows * mat.cols] << std::endl;
			data[i + 2 * mat.rows * mat.cols] = (float)uc_pixel[0] / 255.0;
			//std::cout << (float)uc_pixel[0] << std::endl;
			//std::cout << data[i + 2 * mat.rows * mat.cols] << std::endl;
			uc_pixel += 3;
			++i;
		}
	}
	return true;
}


float* preprocess(std::string image_path, int target_height, int target_width)
{
	float* data; // 用于返回预处理结果
	data = (float*)malloc(sizeof(float) * 3 * target_width * target_height);
	std::vector<cv::Mat> srcImg;
	std::vector<YOLOV5ScaleParams> vetyolovtparams;
	load_images(image_path, srcImg);
	cv::Mat mat_rs;
	YOLOV5ScaleParams scale_params;
	resize_images(srcImg.at(0), mat_rs, target_height, target_width, scale_params);
	vetyolovtparams.push_back(scale_params);
	normalization(mat_rs, data);
	return data;
}

//int main()
//{
//	std::string image_path = "F:\\test\\b1e8eaac1fd5ff5ff338351ac27c5c09.jpg";
//	float* data = preprocess(image_path, 64, 64);
//	std::ofstream outfile("F:\\123.txt");
//	for (int i = 0; i < 3*64*64; i++)
//	{
//		outfile << data[i] << std::endl;
//	}
//	outfile.close();
//}
