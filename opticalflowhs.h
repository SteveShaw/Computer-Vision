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
	void CalcSobel();

	void SetInputTwoImages(unsigned char* ia, unsigned char* ib)
	{
			m_ptrA = ia;
			m_ptrB = ib;
	}


private:
	//Sobel Calc
	std::vector<float> m_Sobel_X0;
	std::vector<float> m_Sobel_X1;
	std::vector<float> m_Sobel_Y0;
	std::vector<float> m_Sobel_Y1;

	std::vector<DP> m_vDP;

	cv::Size m_ImageSize;
	bool m_UsePrev;
	float m_Lambda;
	int m_ImageStep;

	unsigned char* m_ptrA;
	unsigned char* m_ptrB;


protected:

	void CalcFirstLine(const VerStep &vs, std::vector<float>& sobelY, int cur_row, int &cur_idx);
	void CalcMiddleLines(const VerStep &vs, std::vector<float> &sobelY, int cur_row, int &cur_idx);
	void CalcLastLine(const VerStep &vs, std::vector<float> &sobelY, int cur_row, int &cur_idx);

};

#endif // OPTICALFLOWHS_H
