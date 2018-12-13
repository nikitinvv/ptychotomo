#include <cufft.h>

class radonusfft
{
	size_t N;
	size_t Ntheta;
	size_t Nz;
	size_t M;
	float mu;
	
	float2 *f;
	float2 *g;
	float *theta;

	float *x;
	float *y;

	float2 *fde;
	float2 *fdee;


	cufftHandle plan2dfwd;
	cufftHandle plan2dadj;

	cufftHandle plan1d;

public:
	radonusfft(size_t Ntheta, size_t Nz, size_t N);
	~radonusfft();	
	void fwdR(float2 *g, float2 *f, float *theta);
	void adjR(float2 *f, float2 *g, float *theta);

	//wrap for python
        void fwd(float* g, int N0, int N1, int N2, float* f, int N3, int N4, int N5, float* theta, int N7);
        void adj(float* f, int N3, int N4, int N5, float* g, int N0, int N1, int N2, float* theta, int N7);
};

