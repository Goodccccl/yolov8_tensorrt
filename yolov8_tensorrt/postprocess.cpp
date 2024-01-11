#include"postprocess.h"


float iou(Detection box1, Detection box2)
{	
	/* 两个anchor之间的iou得分 */
	float box1_w = box1.box[2] - box1.box[0];
	float box1_h = box1.box[3] - box1.box[1];
	float box1_area = box1_w * box1_h;
	float box2_w = box2.box[2] - box2.box[0];
	float box2_h = box2.box[3] - box2.box[1];
	float box2_area = box2_w * box2_h;
	float x1 = std::max(box1.box[0], box2.box[0]);
	float y1 = std::max(box1.box[1], box2.box[1]);
	float x2 = std::min(box1.box[2], box2.box[2]);
	float y2 = std::min(box1.box[3], box2.box[3]);
	float w = x2 - x1;
	float h = y2 - y1;
	float over_area = w * h;
	float iou_score = over_area / (box1_area + box2_area - over_area);
	return iou_score;
}

static bool sort_score(Detection box1, Detection box2)
{
	return box1.conf > box2.conf ? true : false;
}

std::vector<Detection> nms(std::vector<Detection> outputs_arrange, float threshold)
{
	std::vector<Detection> nms_results;
	std::sort(outputs_arrange.begin(), outputs_arrange.end(), sort_score);	// 用得分进行排序
	while (outputs_arrange.size() > 0)
	{
		nms_results.push_back(outputs_arrange[0]);
		int index = 1;
		while (index < outputs_arrange.size())
		{
			float iou_score = iou(outputs_arrange[0], outputs_arrange[index]);
			if (iou_score > threshold)
			{
				outputs_arrange.erase(outputs_arrange.begin() + index);		// 删除
			}
			else
			{
				index++;
			}
		}
		outputs_arrange.erase(outputs_arrange.begin());
	}
	return nms_results;
}


cv::Mat draw(std::string src_imgPath, std::vector<YOLOV8ScaleParams> vetyolovtparams, std::vector<Detection> nms_result)
{
	cv::Mat img = cv::imread(src_imgPath, 1);
	cv::Mat img_;
	img.copyTo(img_);
	int index = 0;
	while (index < nms_result.size())
	{
		float x1 = nms_result[index].box[0];
		float y1 = nms_result[index].box[1];
		float x2 = nms_result[index].box[2];
		float y2 = nms_result[index].box[3];
		float ratio = 1 / vetyolovtparams[0].r;
		x1 = (x1 - vetyolovtparams[0].dw) * ratio;
		y1 = (y1 - vetyolovtparams[0].dh) * ratio;
		x2 = (x2 - vetyolovtparams[0].dw) * ratio;
		y2 = (y2 - vetyolovtparams[0].dh) * ratio;
		cv::rectangle(img_, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2);
		index++;
	}
	return img_;
}