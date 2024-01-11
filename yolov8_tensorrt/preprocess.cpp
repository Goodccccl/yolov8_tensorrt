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
			cv::cvtColor(srcimg, srcimg, cv::COLOR_BGR2RGB);
			srcImg.push_back(srcimg);
		}
		return true;
	}
}


void resize_images(cv::Mat& mat, cv::Mat& mat_rs, int target_height, int target_width, YOLOV8ScaleParams& scale_params, int& new_w, int& new_h)
{
	if (mat.empty()) return;
	int img_height = static_cast<int>(mat.rows);
	int img_width = static_cast<int>(mat.cols);

	//cv::imshow("mat_rs", mat_rs);
	//cv::waitKey(1000);
	// 获取缩放ratio
	float w_r = float(target_width) / float(img_width);
	float h_r = float(target_height) / float(img_height);
	float r = std::min(w_r, h_r);
	// 得到未填充的缩放尺寸
	int new_unpad_w = static_cast<int>((float)img_width * r);	
	int new_unpad_h = static_cast<int>((float)img_height * r);
	new_w = ceil(float(new_unpad_w) / 32) * 32;
	new_h = ceil(float(new_unpad_h) / 32) * 32;
	int pad_w = new_w - new_unpad_w;
	int pad_h = new_h - new_unpad_h;

	// 创建输出图像
	mat_rs = cv::Mat(new_h, new_w, CV_8UC3, cv::Scalar(114, 114, 114));

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


void resize_images2(cv::Mat& mat, cv::Mat& mat_rs, int target_height, int target_width, YOLOV8ScaleParams& scale_params)
{
	if (mat.empty()) return;
	int img_height = static_cast<int>(mat.rows);
	int img_width = static_cast<int>(mat.cols);

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

	// 创建输出图像
	mat_rs = cv::Mat(target_height, target_width, CV_8UC3, cv::Scalar(114, 114, 114));

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

/* 归一化 && hwc2chw */
bool normalization(cv::Mat mat, float *data)
{
	if (mat.empty()) return false;
	int i = 0;
	for (int row = 0; row < mat.rows; row++) {
		uchar* uc_pixel = mat.data + row * mat.step;
		//std::cout << mat.step << std::endl;
		for (int col = 0; col < mat.cols; col++) {

			data[i] = (float)uc_pixel[0] / 255.0;

			data[i + mat.rows * mat.cols] = (float)uc_pixel[1] / 255.0;

			data[i + 2 * mat.rows * mat.cols] = (float)uc_pixel[2] / 255.0;

			uc_pixel += 3;
			++i;
		}
	}
	return true;
}


float* preprocess(std::string image_path, int target_height, int target_width, std::vector<YOLOV8ScaleParams> &vetyolovtparams)
//float* preprocess(std::string image_path, int target_height, int target_width, int& new_w, int& new_h, std::vector<YOLOV5ScaleParams>& vetyolovtparams)
{
	float* data = (float*)malloc(sizeof(float) * 3 * target_width * target_height);
	std::vector<cv::Mat> srcImg;
	load_images(image_path, srcImg);
	cv::Mat mat_rs;
	YOLOV8ScaleParams scale_params;
	//resize_images(srcImg.at(0), mat_rs, target_height, target_width, scale_params, new_w, new_h);
	resize_images2(srcImg.at(0), mat_rs, target_height, target_width, scale_params);
	//cv::imwrite("D:\\1\\1280.jpg", mat_rs);
	vetyolovtparams.push_back(scale_params);
	normalization(mat_rs, data);
	//std::ofstream outfile("D:\\1\\cpp\\preprocess.txt");
 //   for (int i = 0; i < 3 * target_width * target_height; i++) {
 //       outfile << data[i] << std::endl;
 //   }
 //   outfile.close();
	return data;
}

cv::Mat preprocess2(std::string image_path, int target_height, int target_width)
{
	// 长宽resize到固定target_size
	cv::Mat image = cv::imread(image_path);
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	float ratio = std::min((float)target_height / image.rows, (float)target_width / image.cols);
	int new_height = int(image.rows * ratio);
	int new_width = int(image.cols * ratio);
	cv::resize(image, image, cv::Size(new_width, new_height));
	int padding_height = target_height - new_height;
	int padding_width = target_width - new_width;
	cv::copyMakeBorder(image, image, padding_height / 2, padding_height /2, padding_width / 2, padding_width / 2, cv::BorderTypes::BORDER_CONSTANT, {114, 114, 114});
	//cv::copyMakeBorder(image, image, 0, padding_height, 0, padding_width, cv::BorderTypes::BORDER_CONSTANT, {114, 114, 114});
	image.convertTo(image, CV_32FC3, 1.0f / 255.0f, 0);
	cv::Mat src = cv::dnn::blobFromImage(image);
	return src;
}

cv::Mat preprocess3(std::string image_path, int target_height, int target_width)
{	
	// 长宽均resize到32倍数
	cv::Mat image = cv::imread(image_path);
	float ratio = std::min((float)target_height / image.rows, (float)target_width / image.cols);
	int new_height = int(image.rows * ratio);
	int new_width = int(image.cols * ratio);
	int new_height2 = ceil(float(new_height) / 32) * 32;
	int new_width2 = ceil(float(new_width) / 32) * 32;
	cv::resize(image, image, cv::Size(new_width, new_height));
	int padding_height = new_height2 - new_height;
	int padding_width = new_width2 - new_width;
	cv::copyMakeBorder(image, image, padding_height / 2, padding_height / 2, padding_width / 2, padding_width / 2, cv::BorderTypes::BORDER_CONSTANT, { 114, 114, 114 });
	//cv::copyMakeBorder(image, image, 0, padding_height, 0, padding_width, cv::BorderTypes::BORDER_CONSTANT, {114, 114, 114});
	image.convertTo(image, CV_32FC3, 1.0f / 255.0f, 0);
	cv::Mat src = cv::dnn::blobFromImage(image);
	return src;
}


cv::Mat preprocess4(std::string image_path)
{
	cv::Mat image = cv::imread(image_path);
	//cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	int origin_height = image.size().height;
	int origin_width = image.size().width;

	/***************************** preprocess *****************************/
	// 缩放
	int dst_size = 640;
	float ratio = std::min((float)dst_size / origin_height, (float)dst_size / origin_width);
	int new_height = (int)(origin_height * ratio);
	int new_width = (int)(origin_width * ratio);
	std::cout << new_height << " " << new_width << std::endl;                     // 640 480
	cv::resize(image, image, cv::Size(new_width, new_height));

	// 填充为正方形
	int padding_height = dst_size - new_height;
	int padding_width = dst_size - new_width;
	std::cout << padding_height << " " << padding_width << std::endl;             // 0 160
	// 填充右下角
	cv::copyMakeBorder(image, image, 0, padding_height, 0, padding_width, cv::BorderTypes::BORDER_CONSTANT, { 114, 114, 114 });
	//cv::imshow("0", image);
	//cv::waitKey(0);

	// 转换为float并归一化
	image.convertTo(image, CV_32FC3, 1.0f / 255.0f, 0);

	// hwc -> nchw
	cv::Mat blob = cv::dnn::blobFromImage(image);
	return blob;
}
