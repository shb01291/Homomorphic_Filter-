#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;


void homorphic(Mat f) {
    f.convertTo(f, CV_32F, 1 / 255.f);

    imshow("original", f);

    add(f, 0.1, f);
    log(f, f);
    Mat F;
    dft(f, F, DFT_COMPLEX_OUTPUT);


    Mat filter = Mat::zeros(F.size(), CV_32FC2);

    for (int y = 0; y < filter.rows; y++)for (int x = 0; x < filter.cols; x++) {
        int xx = x > filter.cols / 2 ? x - filter.cols : x;
        int yy = y > filter.rows / 2 ? y - filter.rows : y;

        float Duv = sqrtf(xx * xx + yy * yy);
        float D0 = 3;
        int c = 1;
        float gammaL = 0.4;
        float gammaH = 1.0;


        float Huv = (gammaH - gammaL) * (1 - exp(-Duv * Duv / (D0 * D0))) + gammaL;

        filter.at<Vec2f>(y, x)[0] = Huv;
        filter.at<Vec2f>(y, x)[1] = Huv;

    }
    multiply(F, filter, F);
    Mat g;
    idft(F, g, DFT_SCALE | DFT_REAL_OUTPUT);
    exp(g, g);
    subtract(g, 0.1, g);


    vector<Mat> channels;
    split(g, channels);
    imshow("res", channels[0]);
  
}

int main()
{
    Mat f = imread("C:\\Users\\rkdxo\\OneDrive\\바탕 화면\\HW\\homo1.jpg",0);
    Mat f2 = imread("C:\\Users\\rkdxo\\OneDrive\\바탕 화면\\HW\\homo2.jpg", 0);
    homorphic(f);
    waitKey();
    homorphic(f2);
    waitKey();

    return 0;
}
