#include <stdio.h>
#include <iostream>
#include <time.h>
#include <iomanip>

#include <cmath>

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
//	if(px1 < px2+thickness && px2 < px1+thickness) {
//		printf("%d %d - %d %d",px1,py1,px2,py2);
//		printf("(abs(i-px1) <= thickness || abs(i-px2) <= thickness) : %d",(abs(i-px1) <= thickness || abs(i-px2) <= thickness));
//		printf("(distance(px1,py1,i,j) + distance(px2,py2,i,j) <= distance(px1,py1,px2,py2) + 1.4142135623730951 * thickness) : %d",(distance(px1,py1,i,j) + distance(px2,py2,i,j) <= distance(px1,py1,px2,py2) + 1.4142135623730951 * thickness));
//	}
//Slope infinite 
	if (px1 < px2+thickness && px2 < px1+thickness && (abs(i-px1) <= thickness || abs(i-px2) <= thickness)
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
Point operator - (const Point& obj){
	return {-obj.x, -obj.y};
}
Point operator * (const Point& obj,double s){
	return {obj.x*s , obj.y*s};
}
double distance (const Point& obj1,const Point& obj2){
	return sqrt((obj1.x-obj2.x)*(obj1.x-obj2.x) + (obj1.y-obj2.y)*(obj1.y-obj2.y));
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

unsigned int nx = 1200;
unsigned int ny = 1200;
unsigned int tx = 8;
unsigned int ty = 4;



Point rotate(Point p,double theta) {
	theta = M_PI * theta / 180.0; // convert to radian
	Point p_new;
	p_new.x = p.x*cos(theta) - p.y*sin(theta);
	p_new.y = p.y*cos(theta) + p.x*sin(theta);

	return p_new;
}

Point translate(Point p,Point new_origin) {
	Point p_new;
	p_new.x = p.x - new_origin.x;
	p_new.y = p.y - new_origin.y;

	return p_new;
}


void drawTest(Quad t,int curr_depth,int max_depth,int theta,PixelInfo* window){

	if(curr_depth >= max_depth)
		return;

	dim3 blocks(nx/tx+1,ny/ty+1);
	dim3 threads(tx,ty);

	t.p2.x = t.p1.x + (t.p4.x-t.p1.x)*1.0/3;
	t.p3.x = t.p1.x + (t.p4.x-t.p1.x)*2.0/3;

	t.p2.y = t.p1.y + (t.p4.y-t.p1.y)*1.0/3;
	t.p3.y = t.p1.y + (t.p4.y-t.p1.y)*2.0/3;

	Point mp = translate(rotate(translate(t.p3,t.p2),theta),-t.p2);
	//std::cout << t.p1 << t.p2 << mp << t.p3 << t.p4 << std::endl;

	Line<<<blocks,threads>>>(nx,ny,t.p1.x,t.p1.y,t.p2.x,t.p2.y,1,window);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	Line<<<blocks,threads>>>(nx,ny,t.p2.x,t.p2.y,mp.x,mp.y,1,window);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	Line<<<blocks,threads>>>(nx,ny,mp.x,mp.y,t.p3.x,t.p3.y,1,window);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	Line<<<blocks,threads>>>(nx,ny,t.p3.x,t.p3.y,t.p4.x,t.p4.y,1,window);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	Quad nt1;
	Quad nt2;
	Quad nt3;
	Quad nt4;

	nt1.p1 = t.p1;
	nt1.p4 = t.p2;

	nt2.p1 = t.p2;
	nt2.p4 = mp;

	nt3.p1 = mp;
	nt3.p4 = t.p3;

	nt4.p1 = t.p3;
	nt4.p4 = t.p4;

	drawTest(nt1,curr_depth+1,max_depth,theta,window);
	drawTest(nt2,curr_depth+1,max_depth,theta,window);
	drawTest(nt3,curr_depth+1,max_depth,theta,window);
	drawTest(nt4,curr_depth+1,max_depth,theta,window);

	drawTest(nt1,curr_depth+1,max_depth,-theta,window);
	drawTest(nt2,curr_depth+1,max_depth,-theta,window);
	drawTest(nt3,curr_depth+1,max_depth,-theta,window);
	drawTest(nt4,curr_depth+1,max_depth,-theta,window);

}

int main() {
	
	Point p[3];
	Quad inital_quad[3];

	dim3 blocks(nx/tx+1,ny/ty+1);
	dim3 threads(tx,ty);

	// Alloc Img //
	PixelInfo *img;
	checkCudaErrors(cudaMallocManaged((void **)&img,nx*ny*sizeof(PixelInfo)));

	for(int step=0; step < 3; step++){
		p[step].x = nx/2.0 +  cos(2.0*M_PI/double(3)*step)*nx/2.0;
		p[step].y = ny/2.0 +  sin(2.0*M_PI/double(3)*step)*ny/2.0;
		//printf("%d %lf - %lf %lf\n",step,2*M_PI/double(shape_pts.n)*step,shape_pts.p[step].x,shape_pts.p[step].y );
	}

	inital_quad[0].p1 = p[0];
	inital_quad[0].p4 = p[1];

	inital_quad[1].p1 = p[1];
	inital_quad[1].p4 = p[2];

	inital_quad[2].p1 = p[2];
	inital_quad[2].p4 = p[3];

	int n = 6;
	for(int depth=0;depth<n;depth++) {
		drawTest(inital_quad[0],0,depth,60,img);
		drawTest(inital_quad[1],0,depth,60,img);
		drawTest(inital_quad[2],0,depth,60,img);

		drawTest(inital_quad[0],0,depth,-60,img);
		drawTest(inital_quad[1],0,depth,-60,img);
		drawTest(inital_quad[2],0,depth,-60,img);


		// Save Img //
		std::stringstream ss;
		ss << "./save_folder/Img-" << std::setfill('0') << std::setw(5) << depth << ".jpg";
		
		saveImage(img,nx,ny,ss.str().c_str());
	}
	// Clean Up //
	checkCudaErrors(cudaFree(img));
	return 0;
}