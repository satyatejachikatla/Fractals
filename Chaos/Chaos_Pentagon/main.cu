#include <stdio.h>
#include <iostream>
#include <time.h>
#include <iomanip>
#include <cmath>

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

struct shape {
	vec2 *p;
	int n;
};

__host__ __device__ vec2 operator / (const vec2& obj,double s){
	return {obj.x/s , obj.y/s};
}
__host__ __device__ vec2 operator + (const vec2& obj1,const vec2& obj2){
	return {obj1.x + obj2.x , obj1.y + obj2.y};
}

__host__ __device__ vec2 operator * (const vec2& obj,double s){
	return {obj.x*s , obj.y*s};
}

__host__ __device__ bool operator != (const vec2& obj1,const vec2& obj2){
	return (obj1.x != obj2.x) && (obj1.y != obj2.y);
}

std::ostream& operator << (std::ostream& out ,const vec2& obj){
	out << obj.x << " " << obj.y << " " ;
	return out;
}

unsigned int nx = 600;
unsigned int ny = 600;

#define RND (curand_uniform(local_rand_state))

__global__ void init_seed(vec2* seed,int *last,shape shape_pts,vec2 img_size,curandState *rand_state) {

	int i = threadIdx.x;

	int nx = img_size.x;
	int ny = img_size.y;

	curandState *local_rand_state = &rand_state[i];
	curand_init(1984, i, 0, &rand_state[i]);

	seed[i].x = RND * nx;
	seed[i].y = RND * ny;

	last[i] = -1;

	if (i == 0){
		for(int step=0; step < shape_pts.n; step++){
			shape_pts.p[step].x = nx/2.0 +  cos(2.0*M_PI/double(shape_pts.n)*step)*nx/2.0;
			shape_pts.p[step].y = ny/2.0 +  sin(2.0*M_PI/double(shape_pts.n)*step)*ny/2.0;
			//printf("%d %lf - %lf %lf\n",step,2*M_PI/double(shape_pts.n)*step,shape_pts.p[step].x,shape_pts.p[step].y );
		}
	}
}

__device__ int choice(int n,curandState *local_rand_state){

	int count_10s = 0;
	int temp_n = n;
	while(temp_n){
		count_10s +=1;
		temp_n/=10;
	}

	int rnd = int(RND*pow(10,count_10s))%n;
	//printf("%d\n",rnd );
	return rnd;
}
__global__ void startChaosRound(vec2 *seed,int *last,shape shape_pts,vec2 img_size,PixelInfo *img,curandState *rand_state){
	int i = threadIdx.x;

	int nx = img_size.x;
	int ny = img_size.y;

	curandState *local_rand_state = &rand_state[i];
	const double percent = 0.5;

	vec2 mp;
	vec2 select;

	int rnd = choice(shape_pts.n,local_rand_state);
	while(last[i] == rnd) {
		rnd = choice(shape_pts.n,local_rand_state);
	}
	select = shape_pts.p[rnd];
	last[i] = rnd;

	mp = (select + seed[i]) * percent;

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

	shape shape_pts;
	shape_pts.n=5;
	checkCudaErrors(cudaMalloc((void **)&shape_pts.p, shape_pts.n*sizeof(vec2)));

	vec2 *seed;
	checkCudaErrors(cudaMalloc((void **)&seed, n*sizeof(vec2)));
	int *last;
	checkCudaErrors(cudaMalloc((void **)&last, n*sizeof(int)));

	curandState *d_rand_state;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state, nx*ny*sizeof(curandState)));

	dim3 blocks(1);
	dim3 threads(n);

	init_seed<<<blocks,threads>>>(seed,last,shape_pts,{double(nx),double(ny)},d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	int itter = 500;
	for(int i=0;i<itter;i++) {
 
 		//std::cout << i << std::endl;

		startChaosRound<<<blocks,threads>>>(seed,last,shape_pts,{double(nx),double(ny)},img,d_rand_state);
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
	checkCudaErrors(cudaFree(shape_pts.p));
	return 0;
}