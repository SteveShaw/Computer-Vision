#include "OpticalFlow.h"
#include <iostream>
using namespace std;

void ComputeOpticalFlow(unsigned char* prev, unsigned char* curr, int step,
                        float* vx, float* vy, int vstep)
{
    vstep /= sizeof(vx[0]);
}


OpticalFlowComputing::OpticalFlowComputing(cv::Size win, cv::Size img, int step)
    :m_step(step)
{
    m_GaussX[0] = 1;
    m_GaussY[0] = 1;

    for( int i = 1; i < win.width; i++ )
    {
        m_GaussX[i] = 1;
        for(int j = i - 1; j > 0; j-- )
        {
            m_GaussX[j] += m_GaussX[j - 1];
        }
    }
    for( int i = 1; i < win.height; i++ )
    {
        m_GaussY[i] = 1;
        for( int j = i - 1; j > 0; j-- )
        {
            m_GaussY[j] += m_GaussY[j - 1];
        }
    }

    m_img_size = img;
    m_win_size = win;

    //
    m_II = new DerProduct[win.height*img.width];
    m_WII = new DerProduct[win.height*img.width];

    //
    m_MemX[0] = new float[m_img_size.height];
    m_MemX[1] = new float[m_img_size.height];

    m_MemY[0] = new float[m_img_size.width];
    m_MemY[1] = new float[m_img_size.width];
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

    m_MemX[0][m_img_size.height - 1] = m_MemX[1][m_img_size.height - 1] = CONV( m_ptrA[pixNumber - m_step],m_ptrA[pixNumber], m_ptrA[pixNumber] );
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
    int ver_rad = (m_win_size.height - 1) >> 1;
    int hor_rad = (m_win_size.width - 1) >> 1;

    float *kx = &m_GaussX[hor_rad];
    float *ky = &m_GaussY[ver_rad];


    int pixel_line = -ver_rad;
    int conv_line = 0;
    int buf_addr = -m_img_size.width;

    int buffer_size = m_img_size.width*m_win_size.height;

    while(pixel_line < m_img_size.height)
    {
        if( conv_line < m_img_size.height )
        {
            /*calculate derivatives for line of image */
            int i = conv_line;
            int left = i - 1;
            int cur = i;
            int right = i + 1;

            int memYline = right & 1;

            if( left < 0 )
                left = 0;
            if( right >= m_img_size.height )
                right = m_img_size.height - 1;

            //BufferAddress += BufferWidth;
            buf_addr += m_img_size.width;
            buf_addr -= ((buf_addr >= buffer_size) ? 0xffffffff : 0) & buffer_size;

            int cur_addr = buf_addr;

            int Line1 = left * m_step;
            int Line2 = cur * m_step;
            int Line3 = right * m_step;

            /* Process first pixel */
            int conv_x = CONV( m_ptrA[Line1 + 1], m_ptrA[Line2 + 1], m_ptrA[Line3 + 1] );
            int conv_y = CONV( m_ptrA[Line3], m_ptrA[Line3], m_ptrA[Line3 + 1] );

            int grad_y = conv_y - m_MemY[memYline][0];
            int grad_x = conv_x - m_MemX[1][cur];

            m_MemY[memYline][0] = conv_y;
            m_MemX[1][cur] = conv_x;

            int grad_t = (float) (m_ptrB[Line2] - m_ptrB[Line2]);

            m_II[cur_addr].xx = grad_x * grad_x;
            m_II[cur_addr].xy = grad_x * grad_y;
            m_II[cur_addr].yy = grad_y * grad_y;
            m_II[cur_addr].xt = grad_x * grad_t;
            m_II[cur_addr].yt = grad_y * grad_t;

            cur_addr++;
            /* Process middle of line */
            for(int j = 1; j < m_img_size.width - 1; j++ )
            {
                conv_x = CONV( m_ptrA[Line1 + j + 1], m_ptrA[Line2 + j + 1], m_ptrA[Line3 + j + 1] );
                conv_y = CONV( m_ptrA[Line3 + j - 1], m_ptrA[Line3 + j], m_ptrA[Line3 + j + 1] );

                grad_y = conv_y - m_MemY[memYline][j];
                grad_x = conv_x - m_MemX[(j - 1) & 1][cur];

                m_MemY[memYline][j] = conv_y;
                m_MemX[(j - 1) & 1][cur] = conv_x;

                grad_t = (float) (m_ptrB[Line2 + j] - m_ptrA[Line2 + j]);

                m_II[cur_addr].xx = grad_x * grad_x;
                m_II[cur_addr].xy = grad_x * grad_y;
                m_II[cur_addr].yy = grad_y * grad_y;
                m_II[cur_addr].xt = grad_x * grad_t;
                m_II[cur_addr].yt = grad_y * grad_t;

                cur_addr++;
            }


            /* Process last pixel of line */
            conv_x = CONV( m_ptrA[Line1 + m_img_size.width - 1], m_ptrA[Line2 + m_img_size.width - 1],
                          m_ptrA[Line3 + m_img_size.width - 1] );

            conv_y = CONV( m_ptrA[Line3 + m_img_size.width - 2], m_ptrA[Line3 + m_img_size.width - 1],
                          m_ptrA[Line3 + m_img_size.width - 1] );


            grad_y = conv_y - m_MemY[memYline][m_img_size.width - 1];
            grad_x = conv_x - m_MemX[(m_img_size.width - 2) & 1][cur];

            m_MemY[memYline][m_img_size.width - 1] = conv_y;

            grad_t = (float) (m_ptrB[Line2 + m_img_size.width - 1] - m_ptrA[Line2 + m_img_size.width - 1]);

            m_II[cur_addr].xx = grad_x * grad_x;
            m_II[cur_addr].xy = grad_x * grad_y;
            m_II[cur_addr].yy = grad_y * grad_y;
            m_II[cur_addr].xt = grad_x * grad_t;
            m_II[cur_addr].yt = grad_y * grad_t;
            cur_addr++;

            /* End of derivatives for line */



            /****************************************************************************************/
            /* ---------Calculating horizontal convolution of processed line----------------------- */
            /****************************************************************************************/
            cur_addr -= m_img_size.width;

            /* process first HorRadius pixels */
            for(int j = 0; j < hor_rad; j++ )
            {
                //int jj;

                m_WII[cur_addr].xx = 0;
                m_WII[cur_addr].xy = 0;
                m_WII[cur_addr].yy = 0;
                m_WII[cur_addr].xt = 0;
                m_WII[cur_addr].yt = 0;

                for(int jj = -j; jj <= hor_rad; jj++ )
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
            for(int j = hor_rad; j < m_img_size.width - hor_rad; j++ )
            {
                float k0 = kx[0];

                m_WII[cur_addr].xx = 0;
                m_WII[cur_addr].xy = 0;
                m_WII[cur_addr].yy = 0;
                m_WII[cur_addr].xt = 0;
                m_WII[cur_addr].yt = 0;

                for(int jj = 1; jj <= hor_rad; jj++ )
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
            for(int j = m_img_size.width - hor_rad; j < m_img_size.width; j++ )
            {
                //int jj;

                m_WII[cur_addr].xx = 0;
                m_WII[cur_addr].xy = 0;
                m_WII[cur_addr].yy = 0;
                m_WII[cur_addr].xt = 0;
                m_WII[cur_addr].yt = 0;

                for(int jj = -hor_rad; jj < m_img_size.width - j; jj++ )
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


        if( pixel_line >= 0 )
        {
            int USpace;
            int BSpace;
            //int address;

            if( pixel_line < ver_rad )
                USpace = pixel_line;
            else
                USpace = ver_rad;

            if( pixel_line >= m_img_size.height - ver_rad)
                BSpace = m_img_size.height - pixel_line - 1;
            else
                BSpace = ver_rad;

            int address = ((pixel_line - USpace) % m_win_size.height) * m_img_size.width;


            for(int j = 0; j < m_img_size.width; j++ )
            {
                int addr = address;

                int A1B2 = 0;
                int A2 = 0;
                int B1 = 0;
                int C1 = 0;
                int C2 = 0;

                for(int i = -USpace; i <= BSpace; i++ )
                {
                    A2 += m_WII[addr + j].xx * ky[i];
                    A1B2 += m_WII[addr + j].xy * ky[i];
                    B1 += m_WII[addr + j].yy * ky[i];
                    C2 += m_WII[addr + j].xt * ky[i];
                    C1 += m_WII[addr + j].yt * ky[i];

                    addr += m_img_size.width;
                    addr -= ((addr >= buffer_size) ? 0xffffffff : 0) & buffer_size;
                }
                /****************************************************************************************\
                * Solve Linear System                                                                    *
                \****************************************************************************************/
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
            vx += step;
            vy += step;
        }                       /*for */


        pixel_line++;
        conv_line++;
    }
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


//    const float m2 = 0.3f;


//    for(int y = 0; y < flow.rows; ++y)
//        for(int x = 0; x < flow.cols; ++x)
//        {
//            Point2f f = flow.at<Point2f>(y, x);

//            if (f.x * f.x + f.y * f.y > minVel * minVel)
//            {
//                Point p1 = Point(x, y) * mult;
//                Point p2 = Point(cvRound((x + f.x*m2) * mult), cvRound((y + f.y*m2) * mult));

//                line(cflow, p1, p2, CV_RGB(0, 255, 0));
//                circle(cflow, Point(x, y) * mult, 2, CV_RGB(255, 0, 0));
//            }
//        }

//    rectangle(cflow, (where.tl() + d) * mult, (where.br() + d - Point(1,1)) * mult, CV_RGB(0, 0, 255));
//    namedWindow(name, 1); imshow(name, cflow);

    for(int i = 0;i< rows;++i)
    {
        for(int j = 0;j<cols;++j)
        {
            float x = vx.at<float>(i,j);
            float y = vy.at<float>(i,j);

            float lens = x*x+y*y;
            if(lens>10 && lens < 100)
            {
                cout<<x<<";"<<y<<endl;
                cv::Point p1(j,i);
                cv::Point p2(cvRound(j+x),cvRound(i+y));

                cv::line(output,p1,p2,cv::Scalar(0,255,0));
                cv::circle(output,p1,1,cv::Scalar(255,0,0));
            }

            //float len = (float)sqrt(x*x+y*y);

        }
    }

}
