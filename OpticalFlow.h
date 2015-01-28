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

struct HorStep
{
	int left;
	int mid;
	int right;
};

struct VerStep
{
	int up;
	int mid;
	int down;
};

static const float MinThr = 2.5f;

inline bool IsVerySmall(const float &val)
{
    return abs(val) < 1e-4;
}

void SaveOF(const cv::Mat &vx, const cv::Mat &vy, cv::Mat &output);

void Flow2RGB(const cv::Mat &vx, const cv::Mat &vy, cv::Mat &output);


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

		int m_hor_rad;
		int m_ver_rad;

		int m_buf_size;

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

protected:
		void CalcDerivative(HorStep& hs, VerStep& vs, int cur_memy, int & cur_addr);
		void CalcHorConvolution(int & cur_addr);
		void SolveLinEq(float *vx, float *vy, int cur_line);

};


#endif // OPTICALFLOW_H_INCLUDED
