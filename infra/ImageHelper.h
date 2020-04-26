#pragma once

struct PixelInfo {
	float r;
	float g;
	float b;
};

void saveImagePPM(PixelInfo* fb,unsigned int nx,unsigned int ny,const char* filename);
void saveImage(PixelInfo* fb,unsigned int nx,unsigned int ny,const char* filename);