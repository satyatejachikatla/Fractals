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

std::ostream& operator << (std::ostream& out ,const Point& obj){
	out << obj.x << " " << obj.y << " " ;
	return out;
}

struct Quad {
	Point p1;
	Point p2;
	Point p3;
	Point p4;
};

unsigned int nx = 600;
unsigned int ny = 600;
unsigned int tx = 8;
unsigned int ty = 4;

void drawQuad(Quad t,PixelInfo* window){

	dim3 blocks(nx/tx+1,ny/ty+1);
	dim3 threads(tx,ty);

	t.p1.x *= nx/3;
	t.p1.y *= ny/3;

	t.p2.x *= nx/3;
	t.p2.y *= ny/3;

	t.p3.x *= nx/3;
	t.p3.y *= ny/3;

	t.p4.x *= nx/3;
	t.p4.y *= ny/3;

	//std::cout << t.p1 << t.p2 << t.p3 << t.p4 << std::endl;

	Line<<<blocks,threads>>>(nx,ny,t.p1.x,t.p1.y,t.p2.x,t.p2.y,1,window);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	Line<<<blocks,threads>>>(nx,ny,t.p2.x,t.p2.y,t.p3.x,t.p3.y,1,window);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	Line<<<blocks,threads>>>(nx,ny,t.p3.x,t.p3.y,t.p4.x,t.p4.y,1,window);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	Line<<<blocks,threads>>>(nx,ny,t.p4.x,t.p4.y,t.p1.x,t.p1.y,1,window);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}


void drawSierpinskisCarpet(Quad t,int curr_depth,int max_depth,PixelInfo* window){

	if(curr_depth >= max_depth)
		return;

	drawQuad(t,window);


	for(double i = t.p1.x , ii = 0; ii < 3 ; i+=(t.p2.x-t.p1.x)/3 ,ii+=1 ){
		for(double j = t.p1.y , jj= 0 ; jj < 3 ; j+=(t.p4.y-t.p1.y)/3 ,jj+=1 ){
			if(ii == 1 && jj == 1) {
				continue;
			}

			Quad nt;

			nt.p1.x=i;
			nt.p1.y=j;

			nt.p2.x=i+(t.p2.x-t.p1.x)/3 ;
			nt.p2.y=j ;

			nt.p3.x=i+(t.p2.x-t.p1.x)/3 ;
			nt.p3.y=j+(t.p4.y-t.p1.y)/3 ;

			nt.p4.x=i ;
			nt.p4.y=j+(t.p4.y-t.p1.y)/3 ;

			drawSierpinskisCarpet( nt ,curr_depth+1,max_depth,window);
		}
	}

}

int main() {
	
	Quad inital_quad;
	inital_quad.p1 = {0,0};
	inital_quad.p2 = {3,0};
	inital_quad.p3 = {3,3};
	inital_quad.p4 = {0,3};

	// Alloc Img //
	PixelInfo *img;
	checkCudaErrors(cudaMallocManaged((void **)&img,nx*ny*sizeof(PixelInfo)));


	int n = 7;
	for(int depth=0;depth<n;depth++) {
		drawSierpinskisCarpet(inital_quad,0,depth,img);

		// Save Img //
		std::stringstream ss;
		ss << "./save_folder/Img-" << std::setfill('0') << std::setw(5) << depth << ".jpg";
		
		saveImage(img,nx,ny,ss.str().c_str());
	}
	// Clean Up //
	checkCudaErrors(cudaFree(img));
	return 0;
}