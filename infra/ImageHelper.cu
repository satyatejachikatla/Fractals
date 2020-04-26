#include <ImageHelper.h>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <vector>

void saveImagePPM(PixelInfo* fb,unsigned int nx,unsigned int ny,const char* filename){

	std::ofstream file(filename,std::ofstream::out);
	file << "P3\n" << nx << " " << ny << "\n255\n";
	for (int j = ny-1; j >= 0; j--) {
		for (int i = 0; i < nx; i++) {
			size_t pixel_index = j*nx + i;
			int r = fb[pixel_index].r;
			int g = fb[pixel_index].g;
			int b = fb[pixel_index].b;
			int ir = r%256;
			int ig = g%256;
			int ib = b%256;
			file << ir << " " << ig << " " << ib << "\n";
		}
	}
	file.close();
}

void saveImage(PixelInfo* fb,unsigned int nx,unsigned int ny,const char* filename){

	cv::Mat img(ny,nx,CV_8UC3,cv::Scalar(0,0,0));

	for (int j = ny-1; j >= 0; j--) {
		for (int i = 0; i < nx; i++) {
			size_t pixel_index = j*nx + i;
			int ir = fb[pixel_index].r;
			int ig = fb[pixel_index].g;
			int ib = fb[pixel_index].b;

			img.at<cv::Vec3b>(ny-j-1,i)[0] = ib%256;/*B*/
			img.at<cv::Vec3b>(ny-j-1,i)[1] = ig%256;/*G*/
			img.at<cv::Vec3b>(ny-j-1,i)[2] = ir%256;/*R*/
		}
	}

	cv::imwrite(filename,img);

}