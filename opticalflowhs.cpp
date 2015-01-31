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

	m_Vel_X0.resize(m_ImageSize.width);
	m_Vel_X1.resize(m_ImageSize.width);
	m_Vel_Y0.resize(m_ImageSize.width);
	m_Vel_Y1.resize(m_ImageSize.width);

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

void OpticalFlowHS::CalcSobel(float* vx, float* vy, int vstep)
{
	int cur_row = 0;
	VerStep vs;
	vs.mid = -m_ImageStep;
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

	DoIter(vs,vstep,vx,vy);


}

void OpticalFlowHS::InitializeVelocityVectors(float *vx, float *vy, int vstep)
{
	if(!m_UsePrev)
	{
		for(int i = 0; i < m_ImageSize.height; i++ )
		{
				memset( vx, 0, m_ImageSize.width * sizeof( float ));
				memset( vy, 0, m_ImageSize.width * sizeof( float ));

				vx += vstep;
				vy += vstep;
		}
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
		m_vDP[cur_idx].alpha = 1.0f / (1.0f/m_Lambda+m_vDP[cur_idx].xx+m_vDP[cur_idx].yy);
		++cur_idx;
}

void OpticalFlowHS::CalcMiddleLines(const VerStep& vs, std::vector<float>& sobelY, int cur_row, int &cur_idx)
{
	for(int i = 1; i< m_ImageSize.width-1;++i)
	{
		std::vector<float>& sobelX = m_Sobel_X0;
		if(((i-1)&1)==1) sobelX = m_Sobel_X1;

		float convX = CONV(m_ptrA[vs.up+i+1],m_ptrA[vs.mid+i+1],m_ptrA[vs.down+i+1]);
		float convY = CONV(m_ptrA[vs.down+i-1],m_ptrA[vs.down+i],m_ptrA[vs.down+i+1]);

		float gradY = (convY-sobelY[i])*0.125f;
		float gradX = (convX - sobelX[cur_row])*0.125f;

		sobelX[cur_row] = convX;
		sobelY[i] = convY;

		float gradT = (float)(m_ptrB[vs.mid+i]-m_ptrA[vs.mid+1]);

		m_vDP[cur_idx].xx = gradX * gradX;
		m_vDP[cur_idx].xy = gradX * gradY;
		m_vDP[cur_idx].yy = gradY * gradY;
		m_vDP[cur_idx].xt = gradX * gradT;
		m_vDP[cur_idx].yt = gradY * gradT;
		m_vDP[cur_idx].alpha = 1.0f / (1.0f/m_Lambda+m_vDP[cur_idx].xx+m_vDP[cur_idx].yy);
		++cur_idx;
	}
}

void OpticalFlowHS::CalcLastLine(const VerStep& vs, std::vector<float>& sobelY, int cur_row, int &cur_idx)
{
	float convX = CONV(m_ptrA[vs.up+m_ImageSize.width-1],m_ptrA[vs.mid+m_ImageSize.width-1],m_ptrA[vs.down+m_ImageSize.width-1]);
	float convY = CONV(m_ptrA[vs.down+m_ImageSize.width-2],m_ptrA[vs.down+m_ImageSize.width-1],m_ptrA[vs.down+m_ImageSize.width-1]);

	std::vector<float>& sobelX = m_Sobel_X0;
	if(((m_ImageSize.width-2)&1)==1) sobelX = m_Sobel_X1;

	float gradY = (convY-sobelY[m_ImageSize.width-1])*0.125f;
	float gradX = (convX - sobelX[cur_row])*0.125f;

	sobelY[m_ImageSize.width-1] = convY;

	float gradT = (float)(m_ptrB[vs.mid+m_ImageSize.width-1]-m_ptrA[vs.mid+m_ImageSize.width-1]);

	m_vDP[cur_idx].xt = gradX * gradT;
	m_vDP[cur_idx].yt = gradY * gradT;
	m_vDP[cur_idx].xx = gradX * gradX;
	m_vDP[cur_idx].xy = gradX * gradY;
	m_vDP[cur_idx].yy = gradY * gradY;
	m_vDP[cur_idx].alpha = 1.0f / (1.0f/m_Lambda+m_vDP[cur_idx].xx+m_vDP[cur_idx].yy);
	++cur_idx;
}

void OpticalFlowHS::DoIter(VerStep &vs, int vstep, float* vx, float* vy)
{
			int iter = 0;


			int lastRow = vstep * (m_ImageSize.height - 1);

			float Eps = 0.0;

			int curPos = 0;

			std::vector<float> &velX = m_Vel_X0;
			std::vector<float> &velY = m_Vel_Y0;


			while( true )
			{

					iter++;

					vs.mid = -vstep;

					curPos = 0;


					for( int i = 0; i < m_ImageSize.height; i++ )
					{
						velX = m_Vel_X0;
						velY = m_Vel_Y0;

						if((i&1)==1)
						{
							velX = m_Vel_X1;
							velY = m_Vel_Y1;
						}


							vs.mid += vstep;
							vs.up = vs.mid - ((vs.mid == 0) ? 0 : vstep);
							vs.down = vs.mid + ((vs.mid == lastRow) ? 0 : vstep);

							/* Process first pixel */
							float averX = (vx[vs.mid] + vx[vs.mid + 1] + vx[vs.up] + vx[vs.down]) / 4;
							float averY = (vy[vs.mid] + vy[vs.mid + 1] + vy[vs.up] + vy[vs.down]) / 4;


							velX[0] = averX -	(m_vDP[curPos].xx * averX + m_vDP[curPos].xy * averY + m_vDP[curPos].xt) * m_vDP[curPos].alpha;
							velY[0] = averY - (m_vDP[curPos].xy * averX + m_vDP[curPos].yy * averY + m_vDP[curPos].yt) * m_vDP[curPos].alpha;

							/* update Epsilon */
							if( !m_UseIter )
							{
									float tmp = (float)fabs(vx[vs.mid] - velX[0]);
									Eps = MAX( tmp, Eps );
									tmp = (float)fabs(vy[vs.mid] - velY[0]);
									Eps = MAX( tmp, Eps );
							}
							++curPos;

							/* Process middle of line */
							for(int j = 1; j < m_ImageSize.width - 1; ++j )
							{
								averX = (vx[vs.mid + j - 1] + vx[vs.mid + j + 1] + vx[vs.up + j] + vx[vs.down + j]) / 4;
								averY = (vy[vs.mid + j - 1] + vy[vs.mid + j + 1] + vy[vs.up + j] + vy[vs.down + j]) / 4;
								velX[j] = averX - (m_vDP[curPos].xx * averX + m_vDP[curPos].xy * averY + m_vDP[curPos].xt) * m_vDP[curPos].alpha;
								velY[j] = averY - (m_vDP[curPos].xy * averX + m_vDP[curPos].yy * averY + m_vDP[curPos].yt) * m_vDP[curPos].alpha;

									/* update Epsilon */
									if( !m_UseIter )
									{
											float tmp = (float)fabs(vx[vs.mid + j] - velX[j]);
											Eps = MAX( tmp, Eps );
											tmp = (float)fabs(vy[vs.mid + j] - velY[j]);
											Eps = MAX( tmp, Eps );
									}
									++curPos;
							}

							averX = (vx[vs.mid + m_ImageSize.width-2] + vx[vs.mid + m_ImageSize.width-1]
									+vx[vs.up + m_ImageSize.width-1]+vx[vs.down + m_ImageSize.width-1])/4.0f;

							averY = (vy[vs.mid + m_ImageSize.width-2] + vy[vs.mid + m_ImageSize.width-1]
									+vy[vs.up + m_ImageSize.width-1]+vy[vs.down + m_ImageSize.width-1])/4.0f;

							/* Process last pixel of line */

							velX[m_ImageSize.width - 1] = averX -	(m_vDP[curPos].xx * averX + m_vDP[curPos].xy * averY + m_vDP[curPos].xt) * m_vDP[curPos].alpha;
							velY[m_ImageSize.width - 1] = averY -	(m_vDP[curPos].xy * averX + m_vDP[curPos].yy * averX + m_vDP[curPos].yt) * m_vDP[curPos].alpha;

							/* update Epsilon */
							if( !m_UseIter )
							{
									float tmp = (float)fabs(vx[vs.mid + m_ImageSize.width - 1] - velX[m_ImageSize.width - 1]);
									Eps = MAX( tmp, Eps );
									tmp = (float)fabs(vy[vs.mid + m_ImageSize.width - 1] -  velY[m_ImageSize.width - 1]);
									Eps = MAX( tmp, Eps );
							}

							++curPos;

							/* store new velocity from old buffer to velocity frame */
							if( i > 0 )
							{
									memcpy( &vx[vs.up], &velX[0], m_ImageSize.width * sizeof( float ));
									memcpy( &vy[vs.up], &velY[0], m_ImageSize.width * sizeof( float ));
							}
					}                       /*for */

					/* store new velocity from old buffer to velocity frame */
					memcpy( &vx[m_ImageSize.width * (m_ImageSize.height - 1)],&velX[0], m_ImageSize.width * sizeof( float ));
					memcpy( &vy[m_ImageSize.width * (m_ImageSize.height - 1)],&velY[0], m_ImageSize.width * sizeof( float ));

					if(m_UseIter && (iter==m_IterNum))
					{
						return;
					}

					if(!m_UseIter && (Eps < m_Eps))
					{
						return;
					}
			}
}

