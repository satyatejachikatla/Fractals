#include <stdio.h>
#include <iostream>
#include <time.h>
#include <iomanip>

#include <iostream>
#include <cstdlib>
#include <ctime>

#include <curand_kernel.h>
#include <cudaErrors.h>
#include <ImageHelper.h>

struct vec2 {
	double x;
	double y;
};

struct Triangle {
	vec2 p1;
	vec2 p2;
	vec2 p3;
};

__host__ __device__ vec2 operator / (const vec2& obj,double s){
	return {obj.x/s , obj.y/s};
}
__host__ __device__ vec2 operator + (const vec2& obj1,const vec2& obj2){
	return {obj1.x + obj2.x , obj1.y + obj2.y};
}

std::ostream& operator << (std::ostream& out ,const vec2& obj){
	out << obj.x << " " << obj.y << " " ;
	return out;
}

unsigned int nx = 600;
unsigned int ny = 600;

#define RND (curand_uniform(local_rand_state))

__global__ void init_seed(vec2* seed,vec2 img_size,curandState *rand_state) {

	int i = threadIdx.x;

	int nx = img_size.x;
	int ny = img_size.y;

	curandState *local_rand_state = &rand_state[i];
	curand_init(1984, i, 0, &rand_state[i]);

	seed[i].x = RND * nx;
	seed[i].y = RND * ny;

}

__global__ void startChaosRound(vec2 *seed,vec2 img_size,PixelInfo *img,curandState *rand_state){
	int i = threadIdx.x;

	int nx = img_size.x;
	int ny = img_size.y;

	Triangle t;
	t.p1 = {0,0};
	t.p2 = {double(nx),0};
	t.p3 = {double(nx)/2,double(ny)};

	curandState *local_rand_state = &rand_state[i];

	vec2 mp;
	vec2 select;

	if(RND < 1.0/3) {
		select = t.p1;
	} else if(RND < 2.0/3) {
		select = t.p2;
	} else {
		select = t.p3;
	}
	
	mp = (select + seed[i]) / 2 ;
	int idx = round(mp.y)*nx + round(mp.x);
	if (idx >= nx*ny)
		return ;

	//printf("%d - %lf %lf - %lf %lf - %lf %lf\n",i,seed[i].x,seed[i].y,select.x,select.y,mp.x,mp.y);

	img[idx] = {255,255,255};
	seed[i] = mp;
}

int main() {
	srand(time(NULL));

	//Curand state var for each pixel//
	PixelInfo *img;
	checkCudaErrors(cudaMallocManaged((void **)&img, nx*ny*sizeof(PixelInfo)));

	int n = 100;

	vec2 *seed;
	checkCudaErrors(cudaMalloc((void **)&seed, n*sizeof(vec2)));

	curandState *d_rand_state;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state, nx*ny*sizeof(curandState)));

	dim3 blocks(1);
	dim3 threads(n);

	init_seed<<<blocks,threads>>>(seed,{double(nx),double(ny)},d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	int itter = 100;
	for(int i=0;i<itter;i++) {
 
 		//std::cout << i << std::endl;

		startChaosRound<<<blocks,threads>>>(seed,{double(nx),double(ny)},img,d_rand_state);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		// Save Img //
		std::stringstream ss;
		ss << "./save_folder/Img-" << std::setfill('0') << std::setw(5) << i << ".jpg";

		saveImage(img,nx,ny,ss.str().c_str());
	}

	// Clean Up //
	checkCudaErrors(cudaFree(img));
	checkCudaErrors(cudaFree(seed));
	checkCudaErrors(cudaFree(d_rand_state));
	return 0;
}