#include <iostream>
#include <string>
#include <caffe\caffe.hpp>
#include <caffe\util\io.hpp>
#include <caffe\caffe.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "head.h"  //****** caffe C++ problem solution *******   !!!just be careful
#include "ECO.h"
//#include "dirent.h"

#define USE_VIDEO

//using namespace std;
using namespace caffe;
//using namespace cv;
using namespace eco;

static string WIN_NAME = "ECO-Tracker";

bool gotBB = false;
bool drawing_box = false;
cv::Rect box;
void mouseHandler(int event, int x, int y, int flags, void *param){
	switch (event){
	case CV_EVENT_MOUSEMOVE:
		if (drawing_box){
			box.width = x - box.x;
			box.height = y - box.y;
		}
		break;
	case CV_EVENT_LBUTTONDOWN:
		drawing_box = true;
		box = cv::Rect(x, y, 0, 0);
		break;
	case CV_EVENT_LBUTTONUP:
		drawing_box = false;
		if (box.width < 0){
			box.x += box.width;
			box.width *= -1;
		}
		if (box.height < 0){
			box.y += box.height;
			box.height *= -1;
		}
		gotBB = true;
		break;
	}
}

void drawBox(cv::Mat& image, cv::Rect box, cv::Scalar color, int thick){
	rectangle(image, cvPoint(box.x, box.y), cvPoint(box.x + box.width, box.y + box.height), color, thick);
}

int main()
{
	
	//***********load net prototxt ********************************
	const string proto("F:\\ECO\\ECO\\ECO\\VGG\\imagenet-vgg-m-2048.prototxt");
	const string model("F:\\ECO\\ECO\\ECO\\VGG\\VGG_CNN_M_2048.caffemodel");
	const string mean_file("F:\\ECO\\ECO\\ECO\\VGG\\VGG_mean.binaryproto");
	
#ifdef  USE_VIDEO
	//***********Frame readed****************************************
	cv::Mat frame;
	cv::Rect result;

	//***********reading frome video**********************************
	cv::namedWindow(WIN_NAME);
	cv::VideoCapture capture;
	capture.open("F:\\tmp.avi");
	if (!capture.isOpened())
	{
		std::cout << "capture device failed to open!" << std::endl;
		return -1;
	}

	//**********Register mouse callback to draw the bounding box******
	cvNamedWindow(WIN_NAME.c_str(), CV_WINDOW_AUTOSIZE);
	cvSetMouseCallback(WIN_NAME.c_str(), mouseHandler, NULL);
	
	capture >> frame;
	cv::Mat temp;
	frame.copyTo(temp);
	while (!gotBB)
	{
		drawBox(frame, box, cv::Scalar(0, 0, 255), 1);
		cv::imshow(WIN_NAME, frame);
		temp.copyTo(frame);
		if (cvWaitKey(20) == 27)
			return 1;
	}
	//************** Remove callback  *********************************
	cvSetMouseCallback(WIN_NAME.c_str(), NULL, NULL);
	//box.x = 100; box.y = 150; box.width = 82; box.height = 211;

	ECO Eco(1, proto, model, mean_file);
	Eco.init(frame, box);
	int idx = 0;
	while (idx++<100)
	{
		capture >> frame;
		//capture >> frame;
		//capture >> frame;
		if (frame.empty())
			return -1;
		Eco.process_frame(frame);
		//rectangle(frame, Point(result.x, result.y), Point(result.x + result.width, result.y + result.height), Scalar(0, 255, 255), 1, 8);
		//imshow(WIN_NAME, frame);
		//waitKey(1);
	}
#else
	DIR *dir=nullptr;
	struct dirent *entry=nullptr;
	string path = "F:\\code_tfy\\Matlab\\ECO\\ECO-Caffe\\3 - test\\sequences\\MountainBike\\img";
	//1:  "F:\\Proj\\testdata\\boat1"
	//2: "E:\\flying_tan\\benchmark_results-master\\vot2014\\basketball"
	//3: "F:\\code_tfy\\Matlab\\ECO\\ECO-Caffe\\3 - test\\sequences\\BlurBody\\img"
	//4:  "E:\\flying_tan\\benchmark_results-master\\OTB100\\police"
	//5: "F:\\code_tfy\\Matlab\\ECO\\ECO-Caffe\\3 - test\\sequences\\Crossing\\img"
	//6: "F:\\code_tfy\\Matlab\\ECO\\ECO-Caffe\\3 - test\\sequences\\MountainBike\\img" Car4
	//7: "F:\\Proj\\testdata\\UE_PIC\\car1\\img"
	if ((dir = opendir(path.c_str())) == NULL)
	{
		assert("Error opening \n ");
		return 1;
	}

	ECO eco_tracker(0,proto, model, mean_file);
	size_t  id = 0;
	cv::Mat frame;
	while ((entry = readdir(dir)) != NULL)
	{
		string img_name = entry->d_name;
		if (*(img_name.end() - 1) == 'g')
		{
			frame = cv::imread(path + "\\" + img_name);
			if (id++ == 0)
			{
				cvNamedWindow(WIN_NAME.c_str(), CV_WINDOW_AUTOSIZE);
				cvSetMouseCallback(WIN_NAME.c_str(), mouseHandler, NULL);

				//cv::resize(frame, frame, cv::Size(frame.cols / 2, frame.rows / 2));
				cv::Mat temp;
				frame.copyTo(temp);
				while (!gotBB)
				{
					drawBox(frame, box, cv::Scalar(0, 0, 255), 1);
					imshow(WIN_NAME, frame);
					temp.copyTo(frame);
					if (cvWaitKey(20) == 27)
						return 1;
				}
				//Remove callback  
				cvSetMouseCallback(WIN_NAME.c_str(), NULL, NULL);

				//Convert im0 to grayscale
				cv::Mat im0_gray;
				if (frame.channels() > 1) {
					cvtColor(frame, im0_gray, CV_BGR2GRAY);
				}

				//box.x = 400 - 1; box.y = 48 - 1; box.width = 87; box.height = 319;
				//Initialize LS tracker
				//box.x =182; box.y = 154; box.width = 100; box.height = 78;
				// car 219 246 288 175
				//box.x = 218; box.y = 245; box.width = 288; box.height = 175;
				eco_tracker.init(frame, box); // police 192 154 100 78

			}
			else
			{
				if (img_name == "0013.jpg")
					std::cout << std::endl;

				//cv::resize(frame, frame, cv::Size(frame.cols /2, frame.rows / 2));
				if (frame.empty())
					return -1;
				cv::Mat im_gray;
				//cvtColor(frame, im_gray, CV_BGR2GRAY);
				eco_tracker.process_frame(frame);
			 
				//cv::rectangle(frame, showRect, cv::Scalar(0, 255, 0));
				//cv::imshow(WIN_NAME, frame);
				//cv::waitKey(1);
			}

		}

	}


	closedir(dir);

#endif 

	return 0;

}