#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "OpticalFlow.h"
#include <fstream>

using namespace std;

int main()
{
    //cout << "Hello world!" << endl;

    cv::Mat prev = cv::imread("/home/xubuntu/Downloads/flow/rubic/rubic.0.bmp");
    cv::Mat cur = cv::imread("/home/xubuntu/Downloads/flow/rubic/rubic.1.bmp");


//    cout<<prev.rows<<endl;
//    cout<<prev.cols<<endl;



    cv::Mat prevGray, curGray;
    cv::cvtColor(prev,prevGray,CV_BGR2GRAY);
    cv::cvtColor(cur,curGray,CV_BGR2GRAY);

    cv::Mat vx(prev.rows,prev.cols,CV_32FC1);
    cv::Mat vy(prev.rows,prev.cols,CV_32FC1);

//    //cv::Mat vy(imgSize.width,imgSize.height,CV_32FC1);



    cv::Size winSize(3,3);
    cv::Size imgSize(prevGray.cols,prevGray.rows);

//    cout<<vx.rows<<endl;
//    cout<<vx.cols<<endl;




    OpticalFlowComputing* ofc = new OpticalFlowComputing(cv::Size(5,5), cv::Size(prevGray.cols,prevGray.rows),prevGray.step1());
    ofc->SetInputTwoImages(prevGray.ptr<unsigned char>(), curGray.ptr<unsigned char>());
    ofc->CalFirstLine();
    ofc->DoWork(vx.ptr<float>(),vy.ptr<float>(),vx.step1());

    //cv::Mat output;
    //resize(gray, tmp, gray.size() * mult, 0, 0, INTER_NEAREST);
    //cv::resize(prev,output,prev.size()*8,0,0,cv::INTER_NEAREST);
    SaveOF(vx,vy,prev);

    cv::imwrite("/home/xubuntu/Project/of.png",prev);




    delete ofc;

//    std::ofstream fs("/home/xubuntu/Project/output.dat");
//    if(fs.is_open())
//    {
//        for(int l = 0;l<vx.rows;++l)
//            for(int m = 0;m<vx.cols;++m)
//            {
//                fs << vx.at<float>(m,l)<<";"<<vy.at<float>(m,l)<<endl;
//            }

//    }

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
