#include <iostream>
#include <time.h>
#include <iomanip>

#include <cudaErrors.h>
#include <thrust/complex.h>
#include <ImageHelper.h>

__global__ void Julia(unsigned int nx,unsigned int ny,double centerx,double centery,double scale_x,double scale_y,PixelInfo *img){

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	const int loops = 200;
	const int strength = 30;


	if(i >= nx || j >= ny){
		return;
	}

	double lowx = centerx - scale_x/2;
	double lowy = centery - scale_y/2;

	thrust::complex<double> c = thrust::complex<double>(-1,0);
	thrust::complex<double> z = thrust::complex<double>(lowx+i/double(nx)*scale_x , lowy+j/double(ny)*scale_y);

	double mag;
	for(int l = 0; l < loops ; l++) {

		z = z*z + c;
		mag = norm(z) ;
		if( mag > 4 ) {
			img[j*nx+i].r = (l*strength/2)%256;
			img[j*nx+i].g = (l*strength)%256;
			img[j*nx+i].b = (l*strength/3)%256;
			return;
		}
	}

	img[j*nx+i].r = 256;
	img[j*nx+i].g = 256;
	img[j*nx+i].b = 256;
}


int main() {

	unsigned int nx = 1200;
	unsigned int ny = 600;
	unsigned int tx = 8;
	unsigned int ty = 4;	

	dim3 blocks(nx/tx+1,ny/ty+1);
	dim3 threads(tx,ty);

	// Alloc Img //
	PixelInfo *img;
	checkCudaErrors(cudaMallocManaged((void **)&img,nx*ny*sizeof(PixelInfo)));

	double count = 10000;

	double centerx = 0.618;
	double centery = 0.0001;

	double img_count;
	double reduce;
	for(img_count=0,reduce = 1; reduce < count+1; reduce*=1.2,img_count++ ){
		// Call //
		Julia<<<blocks,threads>>>(nx,ny,centerx,centery,4.0f/reduce,2.0f/reduce,img);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		// Save Img //
		std::stringstream ss;
		ss << "./save_folder/Img-" << std::setfill('0') << std::setw(5) << img_count << ".jpg";
		
		saveImage(img,nx,ny,ss.str().c_str());
	}

	// Clean Up //
	checkCudaErrors(cudaFree(img));
	return 0;
}