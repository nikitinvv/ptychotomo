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
	float2 *ff;
	float2 *gg;	
	float2 *f0;
	float2 *g0;	
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
	void fwdR(float2 *g, float2 *f);
	void adjR(float2 *f, float2 *g);
	void setobjc(float* theta);
	void grad_tomoc(float2* f, float2* g, float eta, int niter);

	//wrap for python
        void fwd(float2* g, int N0, int N1, int N2, float2* f, int N3, int N4, int N5);
		void adj(float2* f, int N3, int N4, int N5, float2* g, int N0, int N1, int N2);
		void grad_tomo(float2* f, int N3, int N4, int N5, float2* g, int N0, int N1, int N2, float eta, int niter);
		void setobj(float* theta, int N7);

};

