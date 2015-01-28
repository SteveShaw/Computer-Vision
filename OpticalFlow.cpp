#include "OpticalFlow.h"
#include <iostream>
using namespace std;



OpticalFlowComputing::OpticalFlowComputing(cv::Size win, cv::Size img, int step)
	:m_step(step)
{

	m_img_size = img;
	m_win_size = win;


	m_GaussX[0] = 1;
	m_GaussY[0] = 1;

	for( int i = 1; i < m_win_size.width; i++ )
	{
		m_GaussX[i] = 1;
		for(int j = i - 1; j > 0; j-- )
		{
			m_GaussX[j] += m_GaussX[j - 1];
		}
	}
	for( int i = 1; i < m_win_size.height; i++ )
	{
		m_GaussY[i] = 1;
		for( int j = i - 1; j > 0; j-- )
		{
			m_GaussY[j] += m_GaussY[j - 1];
		}
	}


	//
	m_II = new DerProduct[m_win_size.height*m_img_size.width];
	m_WII = new DerProduct[m_win_size.height*m_img_size.width];

	//
	m_MemX[0] = new float[m_img_size.height];
	m_MemX[1] = new float[m_img_size.height];

	m_MemY[0] = new float[m_img_size.width];
	m_MemY[1] = new float[m_img_size.width];

	m_ver_rad = (m_win_size.height - 1) >> 1;
	m_hor_rad = (m_win_size.width - 1) >> 1;

	m_buf_size = m_img_size.width*m_win_size.height;
}

void OpticalFlowComputing::CalFirstLine()
{
	m_MemY[0][0] = m_MemY[1][0] = CONV( m_ptrA[0], m_ptrA[0], m_ptrA[1] );
	m_MemX[0][0] = m_MemX[1][0] = CONV( m_ptrA[0], m_ptrA[0], m_ptrA[m_step] );

	for(int j = 1; j < m_img_size.width - 1; j++ )
	{
		m_MemY[0][j] = m_MemY[1][j] = CONV( m_ptrA[j - 1], m_ptrA[j], m_ptrA[j + 1] );
	}

	int pixNumber = m_step;
	for(int i = 1; i < m_img_size.height - 1; i++ )
	{
		m_MemX[0][i] = m_MemX[1][i] = CONV( m_ptrA[pixNumber - m_step],m_ptrA[pixNumber], m_ptrA[pixNumber + m_step] );
		pixNumber += m_step;
	}

	m_MemY[0][m_img_size.width- 1] = m_MemY[1][m_img_size.width - 1] = CONV( m_ptrA[m_img_size.width - 2],
			m_ptrA[m_img_size.width - 1], m_ptrA[m_img_size.width - 1] );

	m_MemX[0][m_img_size.height - 1] = m_MemX[1][m_img_size.height - 1] = CONV( m_ptrA[pixNumber - m_step],m_ptrA[pixNumber],
			m_ptrA[pixNumber] );
}

OpticalFlowComputing::~OpticalFlowComputing()
{
	delete []m_II;
	delete []m_WII;

	delete []m_MemX[0];
	delete []m_MemY[0];

	delete []m_MemX[1];
	delete []m_MemY[1];
}

void OpticalFlowComputing::DoWork(float* vx, float* vy, int step)
{

	int pixel_line = -m_ver_rad;
	int conv_line = 0;
	int buf_addr = -m_img_size.width;


	HorStep hor_step;
	VerStep ver_step;

	while(pixel_line < m_img_size.height)
	{
		if( conv_line < m_img_size.height )
		{
			hor_step.left = conv_line-1;
			hor_step.mid = conv_line;
			hor_step.right = conv_line+1;

			int memY_pos = hor_step.right & 1;

			if( hor_step.left < 0 )
				hor_step.left = 0;
			if( hor_step.right >= m_img_size.height )
				hor_step.right = m_img_size.height - 1;

			//BufferAddress += BufferWidth;
			buf_addr += m_img_size.width;
			buf_addr -= ((buf_addr >= m_buf_size) ? 0xffffffff : 0) & m_buf_size;

			int cur_addr = buf_addr;

			ver_step.up = hor_step.left * m_step;
			ver_step.mid = hor_step.mid * m_step;
			ver_step.down = hor_step.right * m_step;

			CalcDerivative(hor_step,ver_step,memY_pos, cur_addr);

			CalcHorConvolution(cur_addr);

		}


		if( pixel_line >= 0 )
		{
			SolveLinEq(vx,vy,pixel_line);
			vx += step;
			vy += step;
		}

		pixel_line++;
		conv_line++;
	}
}

//cur_my: current memory y line
void OpticalFlowComputing::CalcDerivative(HorStep& hs, VerStep& vs, int cur_memy, int & cur_addr)
{
	/* Process first pixel */
	float conv_x = CONV( m_ptrA[vs.up + 1], m_ptrA[vs.mid + 1], m_ptrA[vs.down + 1] );
	float conv_y = CONV( m_ptrA[vs.down], m_ptrA[vs.down], m_ptrA[vs.down + 1] );

	float grad_y = conv_y - m_MemY[cur_memy][0];
	float grad_x = conv_x - m_MemX[1][hs.mid];

	m_MemY[cur_memy][0] = conv_y;
	m_MemX[1][hs.mid] = conv_x;

	float grad_t = (float) (m_ptrB[vs.mid] - m_ptrA[vs.mid]);

	m_II[cur_addr].xx = grad_x * grad_x;
	m_II[cur_addr].xy = grad_x * grad_y;
	m_II[cur_addr].yy = grad_y * grad_y;
	m_II[cur_addr].xt = grad_x * grad_t;
	m_II[cur_addr].yt = grad_y * grad_t;

	cur_addr++;

	/* Process middle of line */
	for(int j = 1; j < m_img_size.width - 1; j++ )
	{
		conv_x = CONV( m_ptrA[vs.up + j + 1], m_ptrA[vs.mid + j + 1], m_ptrA[vs.down + j + 1] );
		conv_y = CONV( m_ptrA[vs.down + j - 1], m_ptrA[vs.down + j], m_ptrA[vs.down + j + 1] );

		grad_y = conv_y - m_MemY[cur_memy][j];
		grad_x = conv_x - m_MemX[(j - 1) & 1][hs.mid];

		m_MemY[cur_memy][j] = conv_y;
		m_MemX[(j - 1) & 1][hs.mid] = conv_x;

		grad_t = (float) (m_ptrB[vs.mid + j] - m_ptrA[vs.mid + j]);

		m_II[cur_addr].xx = grad_x * grad_x;
		m_II[cur_addr].xy = grad_x * grad_y;
		m_II[cur_addr].yy = grad_y * grad_y;
		m_II[cur_addr].xt = grad_x * grad_t;
		m_II[cur_addr].yt = grad_y * grad_t;

		cur_addr++;
	}


	/* Process last pixel of line */
	conv_x = CONV( m_ptrA[vs.up + m_img_size.width - 1], m_ptrA[vs.mid + m_img_size.width - 1],
			m_ptrA[vs.down + m_img_size.width - 1] );

	conv_y = CONV( m_ptrA[vs.down + m_img_size.width - 2], m_ptrA[vs.down + m_img_size.width - 1],
			m_ptrA[vs.down + m_img_size.width - 1] );


	grad_y = conv_y - m_MemY[cur_memy][m_img_size.width - 1];
	grad_x = conv_x - m_MemX[(m_img_size.width - 2) & 1][hs.mid];

	m_MemY[cur_memy][m_img_size.width - 1] = conv_y;

	grad_t = (float) (m_ptrB[vs.mid + m_img_size.width - 1] - m_ptrA[vs.mid + m_img_size.width - 1]);

	m_II[cur_addr].xx = grad_x * grad_x;
	m_II[cur_addr].xy = grad_x * grad_y;
	m_II[cur_addr].yy = grad_y * grad_y;
	m_II[cur_addr].xt = grad_x * grad_t;
	m_II[cur_addr].yt = grad_y * grad_t;
	cur_addr++;
}

void OpticalFlowComputing::CalcHorConvolution(int &cur_addr)
{


	//Calculating horizontal convolution

	cur_addr -= m_img_size.width;

	/* process first HorRadius pixels */

	float *kx = &m_GaussX[m_hor_rad];

	for(int j = 0; j < m_hor_rad; j++ )
	{

		m_WII[cur_addr].xx = 0;
		m_WII[cur_addr].xy = 0;
		m_WII[cur_addr].yy = 0;
		m_WII[cur_addr].xt = 0;
		m_WII[cur_addr].yt = 0;

		for(int jj = -j; jj <= m_hor_rad; jj++ )
		{
			float kjj = kx[jj];

			m_WII[cur_addr].xx += m_II[cur_addr + jj].xx * kjj;
			m_WII[cur_addr].xy += m_II[cur_addr + jj].xy * kjj;
			m_WII[cur_addr].yy += m_II[cur_addr + jj].yy * kjj;
			m_WII[cur_addr].xt += m_II[cur_addr + jj].xt * kjj;
			m_WII[cur_addr].yt += m_II[cur_addr + jj].yt * kjj;
		}
		cur_addr++;
	}


	/* process inner part of line */
	for(int j = m_hor_rad; j < m_img_size.width - m_hor_rad; j++ )
	{
		float k0 = kx[0];

		m_WII[cur_addr].xx = 0;
		m_WII[cur_addr].xy = 0;
		m_WII[cur_addr].yy = 0;
		m_WII[cur_addr].xt = 0;
		m_WII[cur_addr].yt = 0;

		for(int jj = 1; jj <= m_hor_rad; jj++ )
		{
			float kjj = kx[jj];

			m_WII[cur_addr].xx += (m_II[cur_addr - jj].xx + m_II[cur_addr + jj].xx) * kjj;
			m_WII[cur_addr].xy += (m_II[cur_addr - jj].xy + m_II[cur_addr + jj].xy) * kjj;
			m_WII[cur_addr].yy += (m_II[cur_addr - jj].yy + m_II[cur_addr + jj].yy) * kjj;
			m_WII[cur_addr].xt += (m_II[cur_addr - jj].xt + m_II[cur_addr + jj].xt) * kjj;
			m_WII[cur_addr].yt += (m_II[cur_addr - jj].yt + m_II[cur_addr + jj].yt) * kjj;
		}
		m_WII[cur_addr].xx += m_II[cur_addr].xx * k0;
		m_WII[cur_addr].xy += m_II[cur_addr].xy * k0;
		m_WII[cur_addr].yy += m_II[cur_addr].yy * k0;
		m_WII[cur_addr].xt += m_II[cur_addr].xt * k0;
		m_WII[cur_addr].yt += m_II[cur_addr].yt * k0;

		cur_addr++;
	}
	/* process right side */
	for(int j = m_img_size.width - m_hor_rad; j < m_img_size.width; j++ )
	{
		//int jj;

		m_WII[cur_addr].xx = 0;
		m_WII[cur_addr].xy = 0;
		m_WII[cur_addr].yy = 0;
		m_WII[cur_addr].xt = 0;
		m_WII[cur_addr].yt = 0;

		for(int jj = -m_hor_rad; jj < m_img_size.width - j; jj++ )
		{
			float kjj = kx[jj];

			m_WII[cur_addr].xx += m_II[cur_addr + jj].xx * kjj;
			m_WII[cur_addr].xy += m_II[cur_addr + jj].xy * kjj;
			m_WII[cur_addr].yy += m_II[cur_addr + jj].yy * kjj;
			m_WII[cur_addr].xt += m_II[cur_addr + jj].xt * kjj;
			m_WII[cur_addr].yt += m_II[cur_addr + jj].yt * kjj;
		}
		cur_addr++;
	}
}

void OpticalFlowComputing::SolveLinEq(float *vx, float *vy, int cur_line)
{

	//Solve Linear Equation

	float *ky = &m_GaussY[m_ver_rad];

	int USpace;
	int BSpace;
	//int address;

	if( cur_line  < m_ver_rad )
		USpace = cur_line;
	else
		USpace = m_ver_rad;

	if( cur_line >= m_img_size.height - m_ver_rad)
		BSpace = m_img_size.height - cur_line - 1;
	else
		BSpace = m_ver_rad;

	int mem_pos = ((cur_line - USpace) % m_win_size.height) * m_img_size.width;

	float A1B2 = 0;
	float A2 = 0;
	float B1 = 0;
	float C1 = 0;
	float C2 = 0;

	int j = 0;

	for(j = 0; j < m_img_size.width; j++ )
	{
		int cur_pos = mem_pos;

		A1B2 = 0;
		A2 = 0;
		B1 = 0;
		C1 = 0;
		C2 = 0;


		for(int i = -USpace; i <= BSpace; i++ )
		{
			A2 += m_WII[cur_pos + j].xx * ky[i];
			A1B2 += m_WII[cur_pos + j].xy * ky[i];
			B1 += m_WII[cur_pos + j].yy * ky[i];
			C2 += m_WII[cur_pos + j].xt * ky[i];
			C1 += m_WII[cur_pos + j].yt * ky[i];

			cur_pos += m_img_size.width;
			cur_pos -= ((cur_pos >= m_buf_size) ? 0xffffffff : 0) & m_buf_size;
		}


		{
			float delta = (A1B2 * A1B2 - A2 * B1);

			if( delta )
			{
				/* system is not singular - solving by Kramer method */
				//float deltaX;
				//float deltaY;
				float Idelta = 8 / delta;

				float deltaX = -(C1 * A1B2 - C2 * B1);
				float deltaY = -(A1B2 * C2 - A2 * C1);

				vx[j] = deltaX * Idelta;
				vy[j] = deltaY * Idelta;
			}
			else
			{
				/* singular system - find optical flow in gradient direction */
				float Norm = (A1B2 + A2) * (A1B2 + A2) + (B1 + A1B2) * (B1 + A1B2);

				if( Norm )
				{
					float IGradNorm = 8 / Norm;
					float temp = -(C1 + C2) * IGradNorm;

					vx[j] = (A1B2 + A2) * temp;
					vy[j] = (B1 + A1B2) * temp;

				}
				else
				{
					vx[j] = 0;
					vy[j] = 0;
				}
			}
		}
		/****************************************************************************************\
						* End of Solving Linear System                                                           *
						\****************************************************************************************/
	}                   /*for */
//	vx += vstep;
//	vy += vstep;
}

//save optical flow result

void SaveOF(const cv::Mat &vx, const cv::Mat &vy, cv::Mat &output)
{
	if(vx.rows!=vy.rows || vx.rows!=output.rows)
	{
		return;
	}

	if(vx.cols!=vy.cols || vx.cols!=output.cols)
	{
		return;
	}

	if(vx.type()!=CV_32FC1||vy.type()!=CV_32FC1||output.type()!=CV_8UC3)
	{
		return;
	}

	int rows = vx.rows;
	int cols = vx.cols;

//	float grad_deg_coe = 90.0f/3.14159f;

//	float mean = 0.0, min = 0.0, max = 1000.0f;

	//transform to HSI image

	//	for(int row = 0;row<rows;++row)
	//	{
	//		for(int col = 0;col<cols;++col)
	//		{
	//			float x = vx.at<float>(i,j);
	//			float y = vy.at<float>(i,j);

	//			cv::cartToPolar(vx,vy,)

	//		}
	//	}


	//    const float m2 = 0.3f;


//	for(int y = 0; y < flow.rows; ++y)
//		for(int x = 0; x < flow.cols; ++x)
//		{
//			Point2f f = flow.at<Point2f>(y, x);

//			if (f.x * f.x + f.y * f.y > minVel * minVel)
//			{
//				Point p1 = Point(x, y) * mult;
//				Point p2 = Point(cvRound((x + f.x*m2) * mult), cvRound((y + f.y*m2) * mult));

//				line(cflow, p1, p2, CV_RGB(0, 255, 0));
//				circle(cflow, Point(x, y) * mult, 2, CV_RGB(255, 0, 0));
//			}
//		}

	//    rectangle(cflow, (where.tl() + d) * mult, (where.br() + d - Point(1,1)) * mult, CV_RGB(0, 0, 255));
	//    namedWindow(name, 1); imshow(name, cflow);

	for(int i = 0;i< rows;++i)
	{
		for(int j = 0;j<cols;++j)
		{
			float x = vx.at<float>(i,j);
			float y = vy.at<float>(i,j);

			float lens = x*x+y*y;
			if(lens > MinThr*MinThr)
			{
				cv::Point p1(j,i);
				cv::Point p2(cvRound(j+x),cvRound(i+y));

				cv::line(output,p1,p2,cv::Scalar(0,255,0));
				//cv::circle(output,p1,1,cv::Scalar(255,0,0));
			}

			//float len = (float)sqrt(x*x+y*y);

		}
	}



}


void Flow2RGB(const cv::Mat &vx, const cv::Mat &vy, cv::Mat &output)
{
	cv::Mat ang(vx.rows,vx.cols,CV_32FC1);
	cv::Mat mag(vx.rows,vx.cols,CV_32FC1);


	cv::cartToPolar(vx,vy,mag,ang,true);
	double mag_max, mag_min;
	cv::minMaxLoc(mag,&mag_min,&mag_max);
	cout<<mag_max<<";"<<mag_min<<endl;

	cv::Mat hsv(mag.size(),CV_8UC3);

	unsigned char* pHSV = hsv.ptr<unsigned char>();

	for(int row = 0;row<mag.rows;++row,pHSV+=hsv.step1())
	{
		unsigned char* cur_ptr = pHSV;
		for(int col = 0;col<mag.cols;++col,cur_ptr+=3)
		{
			cur_ptr[0] = cv::saturate_cast<unsigned char>(ang.at<float>(row,col)*90.0f/3.141593);
			float scale_mag = (mag.at<float>(row,col)-mag_min)/(mag_max-mag_min)*255.0f;
			cur_ptr[2] = cv::saturate_cast<unsigned char>(scale_mag);
			cur_ptr[1] = 255;
		}
	}

	//convert from hsv to color;
	cv::cvtColor(hsv,output,cv::COLOR_HSV2BGR);
}
