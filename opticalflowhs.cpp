#include "opticalflowhs.h"


OpticalFlowHS::OpticalFlowHS(cv::Size imgSize, bool usePrev, float lambda, int imgStep)
	:m_ImageSize(imgSize)
	,m_UsePrev(usePrev)
	,m_Lambda(lambda)
	,m_ImageStep(imgStep)
{
	m_Sobel_X0.resize(m_ImageSize.height);
	m_Sobel_X1.resize(m_ImageSize.height);
	m_Sobel_Y0.resize(m_ImageSize.width);
	m_Sobel_Y1.resize(m_ImageSize.width);

	m_vDP.resize(m_ImageSize.width*m_ImageSize.height);
}

OpticalFlowHS::~OpticalFlowHS()
{

}

void OpticalFlowHS::CalcFirstLineSobel()
{
	m_Sobel_Y0[0] = m_Sobel_Y1[0] = CONV( m_ptrA[0], m_ptrA[0], m_ptrA[1] );
	m_Sobel_X0[0] = m_Sobel_X1[0] = CONV( m_ptrA[0], m_ptrA[0], m_ptrA[m_ImageStep] );

	for(int j = 1; j < m_ImageSize.width - 1; j++ )
	{
		m_Sobel_Y0[j] = m_Sobel_Y1[j] = CONV( m_ptrA[j - 1], m_ptrA[j], m_ptrA[j + 1] );
	}

	int cur_line = m_ImageStep;
	for(int i = 1; i < m_ImageSize.height - 1; i++ )
	{
		m_Sobel_X0[i] = m_Sobel_X1[i] = CONV( m_ptrA[cur_line - m_ImageStep],m_ptrA[cur_line], m_ptrA[cur_line + m_ImageStep] );
		cur_line += m_ImageStep;
	}

	m_Sobel_Y0[m_ImageSize.width- 1] = m_Sobel_Y1[m_ImageSize.width - 1] = CONV( m_ptrA[m_ImageSize.width - 2],
			m_ptrA[m_ImageSize.width - 1], m_ptrA[m_ImageSize.width - 1] );

	m_Sobel_X0[m_ImageSize.height - 1] = m_Sobel_X1[m_ImageSize.height - 1] = CONV( m_ptrA[cur_line - m_ImageStep],m_ptrA[cur_line],
			m_ptrA[cur_line] );
}

void OpticalFlowHS::CalcSobel()
{
	int cur_row = 0;
	VerStep vs;
	int last_row = m_ImageStep*(m_ImageSize.height-1);

	int cur_idx = 0;

	while(cur_row<m_ImageSize.height)
	{
		vs.mid += m_ImageStep;
		vs.up = vs.mid - ((vs.mid==0)?0:m_ImageStep);
		vs.down = vs.mid +((vs.mid==last_row)?0:m_ImageStep);



		std::vector<float>& vSobelY = m_Sobel_Y0;

		if(((cur_row+1)&1) == 1)
		{
			vSobelY = m_Sobel_Y1;
		}


		CalcFirstLine(vs,vSobelY, cur_row,cur_idx);
		CalcMiddleLines(vs,vSobelY,cur_row,cur_idx);
		CalcLastLine(vs,vSobelY,cur_row,cur_idx);

		++cur_row;
	}
}

void OpticalFlowHS::CalcFirstLine(const VerStep &vs, std::vector<float>& sobelY, int cur_row, int &cur_idx)
{
		float convX = CONV(m_ptrA[vs.up+1],m_ptrA[vs.mid+1],m_ptrA[vs.down+1]);
		float convY = CONV(m_ptrA[vs.down],m_ptrA[vs.down],m_ptrA[vs.down+1]);


		float gradX = (convX - m_Sobel_X1[cur_row])*0.125f;
		float gradY = (convY - sobelY[0])*0.125f;

		m_Sobel_X1[cur_row] = convX;
		sobelY[0] = convY;

		float gradT = (float)(m_ptrB[vs.mid] - m_ptrA[vs.mid]);

		//Set Derivative Product

		m_vDP[cur_idx].xx = gradX * gradX;
		m_vDP[cur_idx].xy = gradX * gradY;
		m_vDP[cur_idx].yy = gradY * gradY;
		m_vDP[cur_idx].xt = gradX * gradT;
		m_vDP[cur_idx].yt = gradY * gradT;
		m_vDP[cur_idx].alpha = 1.0f / (m_Lambda+m_vDP[cur_idx].xx+m_vDP[cur_idx].yy);
		++cur_idx;
}

void OpticalFlowHS::CalcMiddleLines(const VerStep& vs, std::vector<float>& sobelY, int cur_row, int &cur_idx)
{
	for(int i = 1; i< m_ImageSize.width-1;++i)
	{
		float convX = CONV(m_ptrA[vs.up+i+1],m_ptrA[vs.mid+i+1],m_ptrA[vs.down+i+1]);
		float convY = CONV(m_ptrA[vs.down+i-1],m_ptrA[vs.down+i],m_ptrA[vs.down+i+1]);

		float gradY = (convY-sobelY[i])*0.125f;

		std::vector<float>& sobelX = m_Sobel_X0;
		if(((i-1)&1)==1) sobelX = m_Sobel_X1;

		float gradX = (convX - sobelX[cur_row])*0.125f;

		sobelX[i] = convX;
		sobelY[i] = convY;

		float gradT = (float)(m_ptrB[vs.mid+i]-m_ptrA[vs.mid+1]);

		m_vDP[cur_idx].xx = gradX * gradX;
		m_vDP[cur_idx].xy = gradX * gradY;
		m_vDP[cur_idx].yy = gradY * gradY;
		m_vDP[cur_idx].xt = gradX * gradT;
		m_vDP[cur_idx].yt = gradY * gradT;
		m_vDP[cur_idx].alpha = 1.0f / (m_Lambda+m_vDP[cur_idx].xx+m_vDP[cur_idx].yy);
		++cur_idx;
	}
}

void OpticalFlowHS::CalcLastLine(const VerStep& vs, std::vector<float>& sobelY, int cur_row, int &cur_idx)
{
	float convX = CONV(m_ptrA[vs.up+m_ImageSize.width+1],m_ptrA[vs.mid+m_ImageSize.width+1],m_ptrA[vs.down+m_ImageSize.width+1]);
	float convY = CONV(m_ptrA[vs.down+m_ImageSize.width-2],m_ptrA[vs.down+m_ImageSize.width-1],m_ptrA[vs.down+m_ImageSize.width-1]);

	float gradY = (convY-sobelY[m_ImageSize.width-1])*0.125f;

	std::vector<float>& sobelX = m_Sobel_X0;
	if(((m_ImageSize.width-2)&1)==1) sobelX = m_Sobel_X1;

	float gradX = (convX - sobelX[cur_row])*0.125f;

	sobelY[m_ImageSize.width-1] = convY;

	float gradT = (float)(m_ptrB[vs.mid+m_ImageSize.width-1]-m_ptrA[vs.mid+m_ImageSize.width-1]);

	m_vDP[cur_idx].xt = gradX * gradT;
	m_vDP[cur_idx].yt = gradY * gradT;
	m_vDP[cur_idx].xx = gradX * gradX;
	m_vDP[cur_idx].xy = gradX * gradY;
	m_vDP[cur_idx].yy = gradY * gradY;
	m_vDP[cur_idx].alpha = 1.0f / (m_Lambda+m_vDP[cur_idx].xx+m_vDP[cur_idx].yy);
	++cur_idx;
}

