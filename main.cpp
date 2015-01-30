#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "OpticalFlow.h"
#include <fstream>

using namespace std;

int main()
{
    //cout << "Hello world!" << endl;

		cv::Mat prev = cv::imread("/home/xubuntu/Downloads/flow/rubic/rubic.1.bmp");
		cv::Mat cur = cv::imread("/home/xubuntu/Downloads/flow/rubic/rubic.2.bmp");


//    cout<<prev.rows<<endl;
//    cout<<prev.cols<<endl;



    cv::Mat prevGray, curGray;
    cv::cvtColor(prev,prevGray,CV_BGR2GRAY);
    cv::cvtColor(cur,curGray,CV_BGR2GRAY);

    cv::Mat vx(prev.rows,prev.cols,CV_32FC1);
    cv::Mat vy(prev.rows,prev.cols,CV_32FC1);

		cout<<"vx step="<<vx.step1()<<endl;

//    //cv::Mat vy(imgSize.width,imgSize.height,CV_32FC1);


//    cout<<vx.rows<<endl;
//    cout<<vx.cols<<endl;




		OpticalFlowComputing* ofc = new OpticalFlowComputing(cv::Size(5,5), cv::Size(prevGray.cols,prevGray.rows),prevGray.step1(),false);
    ofc->SetInputTwoImages(prevGray.ptr<unsigned char>(), curGray.ptr<unsigned char>());
    ofc->CalFirstLine();
    ofc->DoWork(vx.ptr<float>(),vy.ptr<float>(),vx.step1());





//		mag.convertTo(mag,-1,1.0/mag_max);


////		//vector<cv::Mat> channels;
//		cv::Mat channels[3];

//		channels[0] = ang;
////		//channels[0].convertTo(channels[0],-1,
//////		//channels.push_back(cv::Mat::ones(ang.size(),CV_32FC1));
//		channels[1] = cv::Mat::ones(ang.size(),CV_32FC1);
////		channels[1].convertTo(channels[1],-1,255.0);
////		cout<<channels[1].at<float>(0,1)<<endl;
//		channels[2] = mag;

////		for(int row = 0;row<mag.rows;++row)
////		{
////			for(int col = 0;col<mag.cols;++col)
////			{
////				float val = mag.at<float>(row,col)/mag_max*4;
////				channels[2].at<float>(row,col) = val>255?255:val;
////			}
////		}

//////		channels[2] = mag;



//		cv::Mat hsv;
//		cv::merge(channels,3,hsv);

//		cv::Mat bgr32F;
//		cv::cvtColor(hsv,bgr32F,cv::COLOR_HSV2BGR);

////		//cout<<is32F<<endl;
		cv::Mat output;


		//bgr32F.convertTo(output,CV_8UC3);
		Flow2RGB(vx,vy,output);
		cv::imwrite("/home/xubuntu/Project/output.png",output);


    //cv::Mat output;
    //resize(gray, tmp, gray.size() * mult, 0, 0, INTER_NEAREST);
    //cv::resize(prev,output,prev.size()*8,0,0,cv::INTER_NEAREST);
		SaveOF(vx,vy,prev);

		cv::imwrite("/home/xubuntu/Project/of.png",prev);




    delete ofc;

//		std::ofstream fs("/home/xubuntu/Project/output.dat");
//		if(fs.is_open())
//		{
//				for(int l = 0;l<vx.rows;++l)
//						for(int m = 0;m<vx.cols;++m)
//						{
//								fs << mag.at<float>(m,l)<<";"<<ang.at<float>(m,l)<<endl;
//						}

//		}

//    fs.close();



    //ofc->DoWork();


//    cv::Mat dest(4,8,CV_32FC1,cv::Scalar(0));

//    dest.at<float>(1,5) = 15.0f;
//    dest.at<float>(0,5) = 10.0f;

//    cout<<"Step="<<dest.step1(0)<<endl;

//    float* data = dest.ptr<float>();


//    cout<<data[5]<<endl;
//    cout<<data[1*dest.step1(0)+5]<<endl;




    return 0;
}
