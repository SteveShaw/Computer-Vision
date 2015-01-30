#ifndef COMMON_H
#define COMMON_H

#define CONV( A, B, C)  ((float)( A +  (B<<1)  + C ))

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



struct DerProduct
{
		float xx;
		float xy;
		float yy;
		float xt;
		float yt;
};

#endif // COMMON_H

