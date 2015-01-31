#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "OpticalFlow.h"
#include "opticalflowhs.h"
#include <fstream>
#include <string>
#include <cstring>

//#include <QDir>
//#include <QDebug>

using namespace std;

static const int mult = 16;

void copnvert2flow(const cv::Mat& velx, const cv::Mat& vely, cv::Mat& flow)
{
		//cv::Mat flow(velx.size(), CV_32FC2);
		for(int y = 0 ; y < flow.rows; ++y)
				for(int x = 0 ; x < flow.cols; ++x)
						flow.at<cv::Point2f>(y, x) = cv::Point2f(velx.at<float>(y, x), vely.at<float>(y, x));
}

void SaveFlow(const cv::Mat& flow, cv::Mat &cflow)
{
	const float m2 = 0.3f;
	const float minVel = 0.1f;

	for(int y = 0; y < flow.rows; ++y)
			for(int x = 0; x < flow.cols; ++x)
			{
					cv::Point2f f = flow.at<cv::Point2f>(y, x);

					if (f.x * f.x + f.y * f.y > minVel * minVel)
					{
							cv::Point p1 = cv::Point(x, y) * mult;
							cv::Point p2 = cv::Point(cvRound((x + f.x*m2) * mult), cvRound((y + f.y*m2) * mult));

							cv::line(cflow, p1, p2, CV_RGB(0, 255, 0));
							cv::circle(cflow, cv::Point(x, y) * mult, 2, CV_RGB(255, 0, 0));
					}
			}

//	return cflow;
}



int main()
{
	//cv::Size szImg(256,240);
	cv::Size szImg(200,200);
	cout<<szImg.width<<endl;
	cout<<szImg.height<<endl;
	getchar();
    //cout << "Hello world!" << endl;

		cv::Mat flow(szImg,CV_32FC2);


		cv::Mat colorFlow(szImg.height*mult, szImg.width*mult, CV_8UC3);



//    cout<<prev.rows<<endl;
//    cout<<prev.cols<<endl;



		cv::Mat vx(szImg.height,szImg.width,CV_32FC1);
		cv::Mat vy(szImg.height,szImg.width,CV_32FC1);

		cv::Mat prevGray(szImg.height,szImg.width,CV_8UC1);
		cv::Mat curGray(szImg.height,szImg.width,CV_8UC1);

		cout<<"vx step="<<vx.step1()<<endl;

//    //cv::Mat vy(imgSize.width,imgSize.height,CV_32FC1);


//    cout<<vx.rows<<endl;
//    cout<<vx.cols<<endl;

		bool UseHS = false;

		string dir = "/home/xubuntu/Downloads/flow/sphere";
		int count = 20;
		char fileName[32];


		if(!UseHS)
		{
			std::cout<<"Using LK Method"<<endl;
			getchar();
			OpticalFlowComputing* ofc = new OpticalFlowComputing(cv::Size(5,5), cv::Size(prevGray.cols,prevGray.rows),prevGray.step1(),false);

			int i = 0;
			while(i<count-1)
			{
				//ofc->InitializeVelocityVectors(vx.ptr<float>(),vy.ptr<float>(),vx.step1());

				string prevPath = dir+"/";
				sprintf(fileName,"sphere.%d.bmp",i);
				prevPath += fileName;

				string curPath = dir+"/";
				sprintf(fileName,"sphere.%d.bmp",i+1);
				curPath += fileName;

				cout<<"Process "<<prevPath<<" and "<<curPath<<endl;

				cv::Mat prevImage = cv::imread(prevPath.c_str());
				cv::Mat curImage = cv::imread(curPath.c_str());

				cv::cvtColor(prevImage,prevGray,CV_BGR2GRAY);
				cv::cvtColor(curImage,curGray,CV_BGR2GRAY);


				ofc->SetInputTwoImages(prevGray.ptr<unsigned char>(), curGray.ptr<unsigned char>());
				ofc->CalFirstLine();
				ofc->DoWork(vx.ptr<float>(),vy.ptr<float>(),vx.step1());


				copnvert2flow(vx,vy,flow);
	//			//cv::Mat colorFlow = SaveFlow(prevGray,flow);
				colorFlow.setTo(cv::Scalar(0,0,0));
				SaveFlow(flow,colorFlow);

				string savePath = dir+"/";
				sprintf(fileName,"LKResult_sphere_%dvs%d.png",i+1,i);
				savePath += fileName;


	//			QString saveFile = savePrefix.append(QString("%1_%2.png").arg(i).arg(i+1));
				cv::imwrite(savePath.c_str(),colorFlow);

				++i;

			}

			delete ofc;
		}

		else
		{
			std::cout<<"Using HS Method"<<endl;
			getchar();
			OpticalFlowHS* ofc = new OpticalFlowHS(szImg,false,0.5,prevGray.step1());
			ofc->SetIterTerm(true,200,0.0);



			int i = 0;
			while(i<count-1)
			{
				ofc->InitializeVelocityVectors(vx.ptr<float>(),vy.ptr<float>(),vx.step1());

				string prevPath = dir+"/";
				sprintf(fileName,"sphere.%d.bmp",i);
				prevPath += fileName;

				string curPath = dir+"/";
				sprintf(fileName,"sphere.%d.bmp",i+1);
				curPath += fileName;

				cout<<"Process "<<prevPath<<" and "<<curPath<<endl;

				cv::Mat prevImage = cv::imread(prevPath.c_str());
				cv::Mat curImage = cv::imread(curPath.c_str());

				cv::cvtColor(prevImage,prevGray,CV_BGR2GRAY);
				cv::cvtColor(curImage,curGray,CV_BGR2GRAY);


				ofc->SetInputTwoImages(prevGray.ptr<unsigned char>(), curGray.ptr<unsigned char>());
				ofc->CalcFirstLineSobel();
				ofc->CalcSobel(vx.ptr<float>(),vy.ptr<float>(),vx.step1());

				copnvert2flow(vx,vy,flow);
	//			//cv::Mat colorFlow = SaveFlow(prevGray,flow);
				colorFlow.setTo(cv::Scalar(0,0,0));
				SaveFlow(flow,colorFlow);

				string savePath = dir+"/";
				sprintf(fileName,"HSResult_sphere_%dvs%d.png",i+1,i);
				savePath += fileName;


	//			QString saveFile = savePrefix.append(QString("%1_%2.png").arg(i).arg(i+1));
				cv::imwrite(savePath.c_str(),colorFlow);

				++i;

			}

			delete ofc;
		}



    return 0;
}
