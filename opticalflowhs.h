#ifndef OPTICALFLOWHS_H
#define OPTICALFLOWHS_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "common.h"



class OpticalFlowHS
{
	//Derivative Product Element
	struct DP
	{
		float xx;
		float yy;
		float xy;
		float xt;
		float yt;
		float alpha;
	};

public:
	OpticalFlowHS(cv::Size imgSize, bool usePrev, float lambda, int imgStep);
	~OpticalFlowHS();

	void CalcFirstLineSobel();
	void CalcSobel(float *vx, float *vy, int vstep);

	void InitializeVelocityVectors(float* vx, float* vy, int vstep);

	void SetInputTwoImages(unsigned char* ia, unsigned char* ib)
	{
			m_ptrA = ia;
			m_ptrB = ib;
	}

	void SetIterTerm(bool useIter, int numIter, double eps)
	{
		m_UseIter = useIter;
		m_IterNum = numIter;
		m_Eps = eps;
	}




private:
	//Sobel Calc
	std::vector<float> m_Sobel_X0;
	std::vector<float> m_Sobel_X1;
	std::vector<float> m_Sobel_Y0;
	std::vector<float> m_Sobel_Y1;

	std::vector<float> m_Vel_X0;
	std::vector<float> m_Vel_X1;
	std::vector<float> m_Vel_Y0;
	std::vector<float> m_Vel_Y1;

	std::vector<DP> m_vDP;

	cv::Size m_ImageSize;
	bool m_UsePrev;
	float m_Lambda;
	int m_ImageStep;

	unsigned char* m_ptrA;
	unsigned char* m_ptrB;

	bool m_UseIter;
	float m_Eps;
	int m_IterNum;


protected:

	void CalcFirstLine(const VerStep &vs, std::vector<float>& sobelY, int cur_row, int &cur_idx);
	void CalcMiddleLines(const VerStep &vs, std::vector<float> &sobelY, int cur_row, int &cur_idx);
	void CalcLastLine(const VerStep &vs, std::vector<float> &sobelY, int cur_row, int &cur_idx);

	void DoIter(VerStep &vs, int vstep, float* vx, float* vy);

};

#endif // OPTICALFLOWHS_H
