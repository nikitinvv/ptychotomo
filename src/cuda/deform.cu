#include "deform.cuh"
#include "kernels.cuh"
#include <stdio.h>
#include <npp.h>
#include "helper_cuda.h"
deform::deform(size_t nz_, size_t n_, size_t ptheta_)
{
    nz = nz_;
    n = n_;
    ptheta = ptheta_;
	cstreams = new cudaStream_t[ptheta];
	nstreams = new NppStreamContext[ptheta];
	for (int i=0;i<ptheta; i++) 
	{
		cudaStreamCreate(&cstreams[i]);
		nstreams[i].hStream=cstreams[i];
	}
}

// destructor, memory deallocation
deform::~deform()
{
    for (int i=0;i<ptheta;i++)
    {
        cudaStreamDestroy(cstreams[i]);
    }
    delete[] cstreams;		
    delete[] nstreams;		
    
}

void deform::remap(size_t g, size_t f, size_t flowx, size_t flowy)
{
	Npp32f *pSrc = (Npp32f *)f;
	NppiSize oSize = {(int)n,(int)nz};
	Npp32f *pDst = (Npp32f *)g;
	NppiRect oROI = {0,0,(int)n,(int)nz};
	int nStep = 4*n;	
	Npp32f *pXMap = (Npp32f *)flowx;
	Npp32f *pYMap = (Npp32f *)flowy;
	int nXMapStep = 4*n;
	int nYMapStep = 4*n;
	for (int i=0;i<ptheta;i++)
	{
		nppiRemap_32f_C1R_Ctx(&pSrc[i*n*nz],oSize,nStep, oROI, &pXMap[i*n*nz], nXMapStep,
			 &pYMap[i*n*nz], nYMapStep, &pDst[i*n*nz], nStep, oSize, NPPI_INTER_LINEAR,
			 nstreams[i]);
		//nppiRemap_32f_C1R (const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp32f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation)
	}
	cudaDeviceSynchronize();
}