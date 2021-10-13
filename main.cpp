#include <iostream>
#include <cstdio>
#include "cudafuns.h"


int main() {

	cv::Mat img = cv::imread("./statue.jpg", CV_BGR2BGRA);
	CudaFuns cudafuns;
	cudafuns.allocateMem();
	cudafuns.justCopy();

	return 0;
}