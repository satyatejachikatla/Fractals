#include <stdio.h>
#include <iostream>
#include <time.h>
#include <iomanip>

#include <cudaErrors.h>
#include <thrust/complex.h>
#include <ImageHelper.h>

__device__ double distance(double x1,double y1,double x2,double y2) {
	return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)); 
}

__global__ void Line(unsigned int nx,unsigned int ny,double px1,double py1,double px2,double py2,double thickness,PixelInfo *img){

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if(i >= nx || j >= ny){
		return;
	}
//Slope infinite 
	if (px1 == px2 && (abs(i-px1) <= thickness || abs(i-px2) <= thickness)
		&&
	    (distance(px1,py1,i,j) + distance(px2,py2,i,j) <= distance(px1,py1,px2,py2) + 1.4142135623730951 * thickness)){
		img[j*nx+i].r = 255;
		img[j*nx+i].g = 255;
		img[j*nx+i].b = 255;
		return;
	} else if (px1 == px2) {
		return;		
	}


// Takes care of all slopes except px2 = px1 case
	double m = 	double(py2-py1)/double(px2-px1);
	// y-y1 = m * (x-x1)
	// y-y2 = m * (x-x2)

	double j_approx1 = m * i - m * px1 + py1;
	double j_approx2 = m * i - m * px2 + py2;

	if (
		(abs(j-j_approx1) <= thickness || abs(j-j_approx2) <= thickness)
		&& 
	    (distance(px1,py1,i,j) + distance(px2,py2,i,j) <= distance(px1,py1,px2,py2) + 1.4142135623730951 * thickness)
	   ){
		img[j*nx+i].r = 255;
		img[j*nx+i].g = 255;
		img[j*nx+i].b = 255;
	} 

}

struct Point {
	double x;
	double y;
};

Point operator / (const Point& obj,double s){
	return {obj.x/s , obj.y/s};
}
Point operator + (const Point& obj1,const Point& obj2){
	return {obj1.x + obj2.x , obj1.y + obj2.y};
}

struct Triangle {
	Point p1;
	Point p2;
	Point p3;	
};

unsigned int nx = 1200;
unsigned int ny = 600;
unsigned int tx = 8;
unsigned int ty = 4;


void drawSierpinskisTriangle(Triangle t,int curr_depth,int max_depth,PixelInfo* window){

	if(curr_depth >= max_depth)
		return;

	dim3 blocks(nx/tx+1,ny/ty+1);
	dim3 threads(tx,ty);

	Line<<<blocks,threads>>>(nx,ny,t.p1.x,t.p1.y,t.p2.x,t.p2.y,1,window);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	Line<<<blocks,threads>>>(nx,ny,t.p2.x,t.p2.y,t.p3.x,t.p3.y,1,window);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	Line<<<blocks,threads>>>(nx,ny,t.p3.x,t.p3.y,t.p1.x,t.p1.y,1,window);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	Point mp1 = (t.p2 + t.p3)/2;
	Point mp2 = (t.p3 + t.p1)/2;
	Point mp3 = (t.p1 + t.p2)/2;

	drawSierpinskisTriangle({t.p1,mp2,mp3},curr_depth+1,max_depth,window);
	drawSierpinskisTriangle({mp1,t.p2,mp3},curr_depth+1,max_depth,window);
	drawSierpinskisTriangle({mp1,mp2,t.p3},curr_depth+1,max_depth,window);

}

int main() {
	
	Triangle inital_triangle;
	inital_triangle.p1 = {0,0};
	inital_triangle.p2 = {double(nx),0};
	inital_triangle.p3 = {double(nx)/2,double(ny)};

	// Alloc Img //
	PixelInfo *img;
	checkCudaErrors(cudaMallocManaged((void **)&img,nx*ny*sizeof(PixelInfo)));


	int n = 10;
	for(int depth=0;depth<n;depth++) {
		drawSierpinskisTriangle(inital_triangle,0,depth,img);

		// Save Img //
		std::stringstream ss;
		ss << "./save_folder/Img-" << std::setfill('0') << std::setw(5) << depth << ".jpg";
		
		saveImage(img,nx,ny,ss.str().c_str());
	}
	// Clean Up //
	checkCudaErrors(cudaFree(img));
	return 0;
}