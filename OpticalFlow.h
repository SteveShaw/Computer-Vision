#ifndef OPTICALFLOW_H_INCLUDED
#define OPTICALFLOW_H_INCLUDED

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "common.h"







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
//    float m_GaussX[16];
//    float m_GaussY[16];

		std::vector<float> m_WeightX;
		std::vector<float> m_WeightY;

		std::vector<float> m_GaussX;
		std::vector<float> m_GaussY;



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

		bool m_UseGauss; //Decide whether the weighting function is Gaussian based weights.



public:
		OpticalFlowComputing(cv::Size win, cv::Size img, int step, bool use_gauss);

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
