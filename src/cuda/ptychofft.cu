#include "ptychofft.cuh"
#include "kernels.cuh"

ptychofft::ptychofft(size_t ntheta, size_t nz, size_t n,
	size_t nscan, size_t detx, size_t dety, size_t nprb, size_t ngpus)
	: ntheta(ntheta), nz(nz), n(n), nscan(nscan), detx(detx), dety(dety), nprb(nprb), ngpus(ngpus) {	
	
	int ffts[2];
	int idist;int odist;
	int inembed[2];int onembed[2];
	ffts[0] = detx; ffts[1] = dety;
	idist = detx*dety; odist = detx*dety;
	inembed[0] = detx; inembed[1] = dety;
	onembed[0] = detx; onembed[1] = dety;
	plan2d = new cufftHandle[ngpus];
	shiftx = new float2*[ngpus];
	shifty = new float2*[ngpus];
	for (int igpu=0;igpu<ngpus;igpu++)
  	{
    	cudaSetDevice(igpu);
		cudaMalloc((void**)&shiftx[igpu],ntheta*nscan*sizeof(float2));
		cudaMalloc((void**)&shifty[igpu],ntheta*nscan*sizeof(float2));
		cufftPlanMany(&plan2d[igpu], 2, ffts, inembed, 1, idist, onembed, 1, odist, CUFFT_C2C, ntheta*nscan); 
	}
	cudaSetDevice(0);
}

ptychofft::~ptychofft(){free();}

void ptychofft::free()
{	
	if (!is_free) 
	{
		for (int igpu=0;igpu<ngpus;igpu++)
  		{
			cudaSetDevice(igpu);
			cudaFree(shiftx[igpu]);
			cudaFree(shifty[igpu]);
			cufftDestroy(plan2d[igpu]);
		}
		is_free = true;   
		cudaSetDevice(0);
	}	
}


void ptychofft::fwd(size_t g_, size_t f_, size_t prb_, size_t scan_, size_t igpu)
{
	cudaSetDevice(igpu);
	
	dim3 BS3d(32,32,1);
	dim3 GS3d0(ceil(nprb*nprb/(float)BS3d.x),ceil(nscan/(float)BS3d.y),ceil(ntheta/(float)BS3d.z));
	dim3 GS3d1(ceil(detx*dety/(float)BS3d.x),ceil(nscan/(float)BS3d.y),ceil(ntheta/(float)BS3d.z));
	dim3 GS2d0(ceil(nscan/(float)BS3d.x),ceil(ntheta/(float)BS3d.y));
	
	float2* f = (float2*)f_;
	float2* g = (float2*)g_;
	float2* prb = (float2*)prb_;
	float* scanx = (float*)&((float*)scan_)[0];
	float* scany = (float*)&((float*)scan_)[ntheta*nscan];
	mul<<<GS3d0,BS3d>>>(g,f,prb,scanx,scany,ntheta,nz,n,nscan,nprb,detx,dety);	
	cufftExecC2C(plan2d[igpu], (cufftComplex*)g,(cufftComplex*)g,CUFFT_FORWARD);
	takeshifts<<<GS2d0,BS3d>>>(shiftx[igpu],shifty[igpu],scanx,scany,ntheta,nscan);		
	shifts<<<GS3d1,BS3d>>>(g, shiftx[igpu], shifty[igpu], ntheta, nscan, detx*dety);
}

void ptychofft::adj(size_t f_, size_t g_, size_t prb_, size_t scan_, size_t igpu)
{
	cudaSetDevice(igpu);
	
	dim3 BS3d(32,32,1);
	dim3 GS3d0(ceil(nprb*nprb/(float)BS3d.x),ceil(nscan/(float)BS3d.y),ceil(ntheta/(float)BS3d.z));
	dim3 GS3d1(ceil(detx*dety/(float)BS3d.x),ceil(nscan/(float)BS3d.y),ceil(ntheta/(float)BS3d.z));
	dim3 GS2d0(ceil(nscan/(float)BS3d.x),ceil(ntheta/(float)BS3d.y));
	
	float2* f = (float2*)f_;
	float2* g = (float2*)g_;
	float2* prb = (float2*)prb_;
	float* scanx = (float*)&((float*)scan_)[0];
	float* scany = (float*)&((float*)scan_)[ntheta*nscan];
	
	takeshifts<<<GS2d0,BS3d>>>(shiftx[igpu],shifty[igpu],scanx,scany,ntheta,nscan);		
	shiftsa<<<GS3d1,BS3d>>>(g, shiftx[igpu], shifty[igpu], ntheta, nscan, detx*dety);
	cufftExecC2C(plan2d[igpu], (cufftComplex*)g,(cufftComplex*)g,CUFFT_INVERSE);
	mula<<<GS3d0,BS3d>>>(f,g,prb,scanx,scany,ntheta,nz,n,nscan,nprb,detx,dety);	
}

void ptychofft::adjprb(size_t prb_, size_t g_, size_t f_, size_t scan_, size_t igpu)
{
	cudaSetDevice(igpu);

	dim3 BS3d(32,32,1);
	dim3 GS3d0(ceil(nprb*nprb/(float)BS3d.x),ceil(nscan/(float)BS3d.y),ceil(ntheta/(float)BS3d.z));
	dim3 GS3d1(ceil(detx*dety/(float)BS3d.x),ceil(nscan/(float)BS3d.y),ceil(ntheta/(float)BS3d.z));
	dim3 GS2d0(ceil(nscan/(float)BS3d.x),ceil(ntheta/(float)BS3d.y));
	
	float2* f = (float2*)f_;
	float2* g = (float2*)g_;
	float2* prb = (float2*)prb_;
	float* scanx = (float*)&((float*)scan_)[0];
	float* scany = (float*)&((float*)scan_)[ntheta*nscan];
	
	takeshifts<<<GS2d0,BS3d>>>(shiftx[igpu],shifty[igpu],scanx,scany,ntheta,nscan);		
	shiftsa<<<GS3d1,BS3d>>>(g, shiftx[igpu], shifty[igpu], ntheta, nscan, detx*dety);
	cufftExecC2C(plan2d[igpu], (cufftComplex*)g,(cufftComplex*)g,CUFFT_INVERSE);
	mulaprb<<<GS3d0,BS3d>>>(f,g,prb,scanx,scany,ntheta,nz,n,nscan,nprb,detx,dety);
}







