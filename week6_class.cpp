#include <iostream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더

using namespace cv;
using namespace std;

double gaussian2D(float c, float r, double sigma);
void myGaussian(const Mat& src_img, Mat& dst_img, Size size);
void myKernelConv(const Mat& src_img, Mat& dst_img, const Mat& kn);
void myMedian(const Mat& src_img, Mat& dst_img, const Size& kn_size);
void doMedianEx();
void myBilateral(const Mat& src_img, Mat& dst_img, int diameter, double sig_r, double sig_s);
void doBilateralEx();
void bilateral(const Mat& src_img, Mat& dst_img, int c, int r, int diameter, double sig_r, double sig_s);
double gaussian(float x, double sigma);
float distance(int x, int y, int i, int j);
void doEdgeEx();
void doCannyEx();
void followEdges(int x, int y, Mat& magnitude, int tUpper, int tLower, Mat& edges);
void edgeDetect(Mat& magnitude, int tUpper, int tLower, Mat& edges);
void nonMaximumSuppression(Mat& magnitudeImage, Mat& directionImge);

int main() {
	doCannyEx();
	return 0;
}
double gaussian2D(float c, float r, double sigma)
{
	return exp(-(pow(c, 2) + pow(r, 2)) / (2 * pow(sigma, 2)))
		/ (2 * CV_PI * pow(sigma, 2));
}

void myGaussian(const Mat& src_img, Mat& dst_img, Size size)
{
	Mat kn = Mat::zeros(size, CV_32FC1);
	double sigma = 1.0;
	float* kn_data = (float*)kn.data;
	for (int c = 0; c < kn.cols; c++) {
		for (int r = 0; r < kn.rows; r++) {
			kn_data[r * kn.cols + c] =
				(float)gaussian2D((float)(c - kn.cols / 2),
					(float)(r - kn.rows / 2), sigma);
		}
	}

	myKernelConv(src_img, dst_img, kn);
}

void myKernelConv(const Mat& src_img, Mat& dst_img, const Mat& kn)
{
	dst_img = Mat::zeros(src_img.size(), CV_8UC1);

	int wd = src_img.cols; int hg = src_img.rows;
	int kwd = kn.cols; int khg = kn.rows;
	int rad_w = kwd / 2; int rad_h = khg / 2;

	float* kn_data = (float*)kn.data;
	uchar* src_data = (uchar*)src_img.data;
	uchar* dst_data = (uchar*)dst_img.data;

	float wei, tmp, sum;

	for (int c = rad_w + 1; c < wd - rad_w; c++) {
		for (int r = rad_h + 1; r < hg - rad_h; r++) {
			tmp = 0.f;
			sum = 0.f;
			for (int kc = -rad_w; kc <= rad_w; kc++) {
				for (int kr = -rad_h; kr <= rad_h; kr++) {
					wei = (float)kn_data[(kr + rad_h) * kwd + (kc + rad_w)];
					tmp += wei * (float)src_data[(r + kr) * wd + (c + kc)];
					sum += wei;
				}
			}
			if (sum != 0.f) tmp = abs(tmp) / sum;
			else tmp = abs(tmp);

			if (tmp > 255.f) tmp = 255.f;

			dst_data[r * wd + c] = (uchar)tmp;
		}
	}
}

void myMedian(const Mat& src_img, Mat& dst_img, const Size& kn_size)
{
	dst_img = Mat::zeros(src_img.size(), CV_8UC1);

	int wd = src_img.cols; int hg = src_img.rows;
	int kwd = kn_size.width; int khg = kn_size.height;
	int rad_w = kwd / 2; int rad_h = khg / 2;

	uchar* src_data = (uchar*)src_img.data;
	uchar* dst_data = (uchar*)dst_img.data;

	float* table = new float[kwd * khg]();
	float tmp;

	for (int c = rad_w + 1; c < wd - rad_w; c++) {
		for (int r = rad_h + 1; r < hg - rad_h; r++) {
			tmp = 0.f;
			for (int kc = -rad_w; kc <= rad_w; kc++) {
				for (int kr = -rad_h; kr <= rad_h; kr++) {
					tmp = (float)src_data[(r + kr) * wd + (c + kc)]; 
					table[(kr + rad_h) * kwd + (kc + rad_w)] = tmp;

				}
			}
			sort(table, table + (kwd * khg)); 
			float med = table[kwd * khg / 2]; 
			dst_data[r * wd + c] = (uchar)med; 
		}
	}

	delete table;
}

void doMedianEx()
{
	cout << "--- doMedianEx() ---\n" << endl;

	Mat src_img = imread("salt_pepper.png", 0);
	if (!src_img.data) printf("No image data \n");

	Mat dst_img;

	myMedian(src_img, dst_img, Size(3, 3));

	Mat result_img;
	hconcat(src_img, dst_img, result_img);
	imshow("doMedianEx()", result_img);
	waitKey(0);
}

void myBilateral(const Mat& src_img, Mat& dst_img, int diameter, double sig_r, double sig_s)
{
	dst_img = Mat::zeros(src_img.size(), CV_8UC1);

	Mat guide_img = Mat::zeros(src_img.size(), CV_64F);
	int wh = src_img.cols; int hg = src_img.rows;
	int radius = diameter / 2;

	for (int c = radius + 1; c < hg - radius; c++) {
		for (int r = radius + 1; r < wh - radius; r++) {
			bilateral(src_img, guide_img, c, r, diameter, sig_r, sig_s);
		}
	}
	guide_img.convertTo(dst_img, CV_8UC1);
}

void doBilateralEx()
{
	cout << "--- doBilateralEx() --- \n" << endl;
	Mat src_img = imread("rock.png", 0);
	Mat dst_img;
	if (!src_img.data) printf("No image data \n");

	myBilateral(src_img, dst_img, 5, 25.0, 50.0);

	Mat result_img;
	hconcat(src_img, dst_img, result_img);
	imshow("doBilateralEx()", result_img);
	waitKey(0);
}

void bilateral(const Mat& src_img, Mat& dst_img, int c, int r, int diameter, double sig_r, double sig_s)
{
	int radius = diameter / 2;

	double gr, gs, wei;
	double tmp = 0;
	double sum = 0;

	for (int kc = -radius; kc <= radius; kc++) {
		for (int kr = -radius; kr <= radius; kr++) {
			gr = gaussian((float)src_img.at<uchar>(c + kc, r + kr) - (float)src_img.at<uchar>(c, r), sig_r);
			gs = gaussian(distance(c, r, c + kc, r + kr), sig_s);
			wei = gr * gs;
			tmp += src_img.at<uchar>(c + kc, r + kr) * wei;
			sum += wei;
		}
	}
	dst_img.at<double>(c, r) = tmp / sum;
}

double gaussian(float x, double sigma)
{
	return exp(-(pow(x, 2) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2)));
}

float distance(int x, int y, int i, int j)
{
	return float(sqrt(pow(x - i, 2) + pow(y - j, 2)));
}

void doEdgeEx()
{
	cout << "--- doEdgeEx() ---\n" << endl;

	Mat src_img = imread("rock.png", 0);
	if (!src_img.data) printf("No image data \n");

	Mat blur_img;
	myGaussian(src_img, blur_img, Size(5, 5));

	float kn_data[] = { 1.f, 0.f, -1.f,
						1.f, 0.f, -1.f,
						1.f, 0.f, -1.f };
	Mat kn(Size(3, 3), CV_32FC1, kn_data);

	cout << "Edge kernel: \n" << kn << endl;

	Mat dst_img;
	myKernelConv(blur_img, dst_img, kn);

	Mat result_img;
	hconcat(src_img, dst_img, result_img);
	imshow("doEdgeEx()", result_img);
	waitKey(0);
}

void doCannyEx()
{
	cout << "--- doCannyEx() ---\n" << endl;

	Mat src_img = imread("rock.png", 0);
	if (!src_img.data) printf("No image data \n");

	Mat dst_img;
	Mat blur_img;
	GaussianBlur(src_img, blur_img, Size(3, 3), 1.5);

	Mat magX = Mat(src_img.rows, src_img.cols, CV_32F);
	Mat magY = Mat(src_img.rows, src_img.cols, CV_32F);
	Sobel(blur_img, magX, CV_32F, 1, 0, 3);
	Sobel(blur_img, magY, CV_32F, 0, 1, 3);

	Mat sum = Mat(src_img.rows, src_img.cols, CV_64F);
	Mat prodX = Mat(src_img.rows, src_img.cols, CV_64F);
	Mat prodY = Mat(src_img.rows, src_img.cols, CV_64F);
	multiply(magX, magX, prodX);
	multiply(magY, magY, prodY);
	sum = prodX + prodY;
	sqrt(sum, sum);

	Mat magnitude = sum.clone();

	Mat slopes = Mat(src_img.rows, src_img.cols, CV_32F);
	divide(magY, magX, slopes);
	nonMaximumSuppression(magnitude, slopes);

	cout << "--- doCannyEx() ---\n" << endl;

	edgeDetect(magnitude, 200, 100, dst_img);
	cout << "--- doCannyEx() ---\n" << endl;
	dst_img.convertTo(dst_img, CV_8UC1);

	cout << "--- doCannyEx() ---\n" << endl;

	Mat result_img;
	hconcat(src_img, dst_img, result_img);
	imshow("doCannyEx()", result_img);
	waitKey(0);
}

void followEdges(int x, int y, Mat& magnitude, int tUpper, int tLower, Mat& edges)
{
	edges.at<float>(y, x) = 255;

	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			if ((i != 0) && (j != 0) && (x + i >= 0) && (y + j >= 0) &&
				(x + i <= magnitude.cols) && (y + j <= magnitude.rows)) {

				if ((magnitude.at<float>(y + j, x + i) > tLower) &&
					(edges.at<float>(y + j, x + i) != 255)) {
					followEdges(x + i, y + j, magnitude, tUpper, tLower, edges);
				}
			}
		}
	}
}

void edgeDetect(Mat& magnitude, int tUpper, int tLower, Mat& edges)
{
	int rows = magnitude.rows;
	int cols = magnitude.cols;

	edges = Mat(magnitude.size(), CV_32F, 0.0);

	for (int x = 0; x < cols; x++) {
		for (int y = 0; y < rows; y++) {
			if (magnitude.at<float>(y, x) >= tUpper) {
				followEdges(x, y, magnitude, tUpper, tLower, edges);
			}
		}
	}
	
}

void nonMaximumSuppression(Mat& magnitudeImage, Mat& directionImge)
{
	Mat checkImage = Mat(magnitudeImage.rows, magnitudeImage.cols, CV_8U);

	MatIterator_<float>itMag = magnitudeImage.begin<float>();
	MatIterator_<float>itDirection = directionImge.begin<float>();
	MatIterator_<unsigned char>itRet = checkImage.begin<unsigned char>();
	MatIterator_<float>itEnd = magnitudeImage.end<float>();

	for (; itMag != itEnd; ++itDirection, ++itRet, ++itMag) {
		const Point pos = itRet.pos();

		float currentDirection = atan(*itDirection) * (180 / 3.142);
		while (currentDirection < 0) currentDirection += 180;

		*itDirection = currentDirection;

		if (currentDirection > 22.5 && currentDirection <= 67.5) {
			if (pos.y > 0 && pos.x > 0 && *itMag <= magnitudeImage.at<float>(pos.y - 1, pos.x - 1)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
			if (pos.y < magnitudeImage.rows - 1 && pos.x < magnitudeImage.cols - 1 && *itMag <= magnitudeImage.at<float>(pos.y + 1, pos.x + 1)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
		}
		else if (currentDirection > 67.5 && currentDirection <= 112.5) {
			if (pos.y > 0 && *itMag <= magnitudeImage.at<float>(pos.y - 1, pos.x)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
			if (pos.y < magnitudeImage.rows - 1 && *itMag <= magnitudeImage.at<float>(pos.y + 1, pos.x)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
		}
		else if (currentDirection > 112.5 && currentDirection <= 157.5) {
			if (pos.y > 0 && pos.x <magnitudeImage.cols -1 && *itMag <= magnitudeImage.at<float>(pos.y - 1, pos.x+1)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
			if (pos.y < magnitudeImage.rows - 1 && pos.x> 0 && *itMag <= magnitudeImage.at<float>(pos.y + 1, pos.x-1)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
		}
		else {
			if (pos.x > 0 && *itMag <= magnitudeImage.at<float>(pos.y, pos.x-1)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
			if (pos.x < magnitudeImage.rows - 1 && *itMag <= magnitudeImage.at<float>(pos.y, pos.x+1)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
		}
	}
}
