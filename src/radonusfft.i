/*interface*/
%module radonusfft

%{
#define SWIG_FILE_WITH_INIT
#include "radonusfft.cuh"
%}

%include "numpy.i"

%init %{
import_array();
%}

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

	float2 *fd;

	cufftHandle plan2dfwd;
	cufftHandle plan2dadj;

	cufftHandle plan1d;
        
public:
	radonusfft(size_t Nz, size_t Ntheta, size_t N);
	~radonusfft();	
        void fwdR(float2 *g, float2 *f, float *theta);
        void adjR(float2 *f, float2 *g, float *theta);

//wrap for python
%apply (float *INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {(float *g, int N0, int N1, int N2)};
%apply (float *IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(float *f, int N3, int N4, int N5)};
%apply (float *IN_ARRAY1, int DIM1) {(float *theta, int N7)};
        void fwd(float* g, int N0, int N1, int N2, float* f, int N3, int N4, int N5, float* theta, int N7);
%clear (float *g, int N0, int N1, int N2);
%clear (float *f, int N3, int N4, int N5);
%clear (float *theta, int N7);

%apply (float *IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(float *g, int N0, int N1, int N2)};
%apply (float *INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {(float *f, int N3, int N4, int N5)};
%apply (float *IN_ARRAY1, int DIM1) {(float *theta, int N7)};
        void adj(float* f, int N3, int N4, int N5, float* g, int N0, int N1, int N2, float* theta, int N7);
%clear (float *g, int N0, int N1, int N2);
%clear (float *f, int N3, int N4, int N5);
%clear (float *theta, int N7);

};


