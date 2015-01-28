#ifndef OPTICALFLOW_H_INCLUDED
#define OPTICALFLOW_H_INCLUDED

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#define CONV( A, B, C)  ((float)( A +  (B<<1)  + C ))

struct DerProduct
{
    float xx;
    float xy;
    float yy;
    float xt;
    float yt;
};

static const float MinThr = 0.1f;

inline bool IsVerySmall(const float &val)
{
    return abs(val) < 1e-4;
}

void SaveOF(const cv::Mat &vx, const cv::Mat &vy, cv::Mat &output);


class OpticalFlowComputing
{
    cv::Size m_win_size;
    cv::Size m_img_size;

    /* Gaussian separable kernels */
    float m_GaussX[16];
    float m_GaussY[16];


    float* m_MemX[2];
    float* m_MemY[2];

    DerProduct *m_II;
    DerProduct *m_WII;//weighted

    int m_step;

    unsigned char* m_ptrA;
    unsigned char* m_ptrB;


public:
    OpticalFlowComputing(cv::Size win, cv::Size img, int step);

    void CalFirstLine();

    void SetInputTwoImages(unsigned char* ia, unsigned char* ib)
    {
        m_ptrA = ia;
        m_ptrB = ib;
    }

    virtual ~OpticalFlowComputing();

    void DoWork(float *vx, float *vy, int step);

};

void ComputeOpticalFlow(unsigned char* prev, unsigned char* curr, int step,
                        float* vx, float* vy, int vstep);

#endif // OPTICALFLOW_H_INCLUDED