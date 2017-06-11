// Example showing how to read and write images
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv/cvaux.hpp"

void ExerciseFive()
{
    // Load an image from file - change this based on your image name
    cv::Mat one = cv::imread("one.jpg");
    cv::Mat two = cv::imread("two.jpg");
    cv::Mat small = cv::imread("small.jpg");
    
    int rows = small.rows;
    int cols = small.cols;
    int channels = small.channels();
    
    cv::Mat roi = one(cv::Range(0, rows), cv::Range(0, cols));
    
    cv::Mat img2gray;
    cv::cvtColor(small, img2gray, cv::COLOR_BGR2GRAY);
    
    cv::Mat mask;
    cv::threshold(img2gray, mask, 220, 255, cv::THRESH_BINARY_INV);
    
    cv::Mat mask_inv;
    cv::bitwise_not(mask, mask_inv);
    
    cv::Mat one_bg;
    cv::bitwise_and(roi, roi, one_bg, mask_inv);
    
    cv::Mat two_fg;
    cv::bitwise_and(small, small, two_fg, mask);
    
    cv::Mat dst;
    cv::add(one_bg, two_fg, dst);
    
    dst.copyTo(roi);
}

void ExerciseSix()
{
    using namespace cv;
    
    Mat img =imread("bookpage.jpg");
    
    cv::Mat threshold;
    cv::threshold(img, threshold, 12, 255, cv::THRESH_BINARY);
    
    Mat grayScaled;
    cv::cvtColor(img, grayScaled, COLOR_BGR2GRAY);
    
    cv::Mat thresholdGray;
    cv::threshold(grayScaled, thresholdGray, 12, 255, cv::THRESH_BINARY);
    
    Mat gaus;
    adaptiveThreshold(grayScaled, gaus, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 115, 1);

}

void ExerciseEight()
{
    using namespace cv;
    
    VideoCapture cap(0);
    
    while(true)
    {
        Mat frame;
        
        cap >> frame;
        
        Mat hsv;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        
        Mat mask;
        inRange(hsv, Scalar(150, 100, 0), Scalar(180, 255, 255), mask);
        
        Mat res;
        bitwise_and(frame, frame, res, mask);
        
        Mat kernel;
        int kernel_size = 15;
        
        kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / float(kernel_size * kernel_size);
        
        Mat smooth;
        filter2D(res, smooth, -1, kernel);
        
        Mat blur;
        GaussianBlur(res, blur, Size(15, 15), 0);
        
        Mat median;
        medianBlur(res, median, 15);
        
        Mat bilateral;
        bilateralFilter(res, bilateral, 15, 75, 75);
        
        imshow("frame", frame);
        //imshow("mask", mask);
        imshow("result", bilateral);
    }
    
}

void ExerciseNine()
{
    using namespace cv;
    
    VideoCapture cap(0);
    
    while(true)
    {
        Mat frame;
        
        cap >> frame;
        
        Mat hsv;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        
        Mat mask;
        inRange(hsv, Scalar(150, 100, 0), Scalar(180, 255, 255), mask);
        
        Mat res;
        bitwise_and(frame, frame, res, mask);
        
        int kernel_size = 5;
        Mat kernel;
        
        kernel = Mat::ones(kernel_size, kernel_size, CV_8U);
        
        Mat erosion;
        erode(mask, erosion, kernel, Point(-1, -1), 1, 1, 1);
        
        Mat dilation;
        dilate(mask, dilation, kernel, Point(-1, -1), 1, 1, 1);
        
        Mat opening;
        morphologyEx(mask, opening, MORPH_OPEN, kernel);
        
        Mat closing;
        morphologyEx(mask, closing, MORPH_CLOSE, kernel);
        
        imshow("frame", frame);
        imshow("mask", mask);
        imshow("opening", opening);
        imshow("closing", closing);
        
        int result = waitKey(5);
        
        if (result == 0x71)
        {
            break;
        }
    }
}

void ExerciseTen()
{
    using namespace cv;
    
    VideoCapture cap(0);
    
    while(true)
    {
        Mat frame;
        
        cap >> frame;
        
        Mat hsv;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        
        
        Mat laplacian;
        Laplacian(frame, laplacian, CV_64F);
        
        Mat sobelx;
        Sobel(frame, sobelx, CV_64F, 1, 0, 5);
        
        Mat sobely;
        Sobel(frame, sobely, CV_64F, 0, 1, 5);
        
        Mat edges;
        Canny(frame, edges, 100, 200);
        
        /*sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
         sobely = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)
         
         cv2.imshow('Original',frame)
         cv2.imshow('Mask',mask)
         cv2.imshow('laplacian',laplacian)
         cv2.imshow('sobelx',sobelx)
         cv2.imshow('sobely',sobely)*/
        
        imshow("Original",frame);
        imshow("laplacian", laplacian);
        imshow("sobelx", sobelx);
        imshow("sobely", sobely);
        imshow("edges", edges);
        
        int result = waitKey(5);
        
        if (result == 0x71)
        {
            break;
        }
    }
}

void ExerciseEleven()
{
    using namespace cv;
    
    Mat img_bgr;
    img_bgr = imread("opencv-template-matching-python-tutorial.jpg");
    
    Mat img_gray;
    cvtColor(img_bgr, img_gray, COLOR_BGR2GRAY);
    
    Mat img_template;
    img_template = imread("opencv-template-for-matching.jpg", 0);
    
    int w = img_template.cols;
    int h = img_template.rows;
    
    Mat res;
    matchTemplate(img_gray, img_template, res, TM_CCOEFF_NORMED);
    
    float threshold = 0.90;
    
    printf("%d, %d\n", res.type(), res.depth());
    
    for (int y = 0; y < res.rows; y++)
    {
        for (int x = 0; x < res.cols; x++)
        {
            float val = res.at<float>(y, x);
            
            if (val > threshold)
            {
                printf("%d, %d, %f\n", x, y, val);
                
                Point p(x, y);
                Point q = p + Point(w, h);
                rectangle(img_bgr, p, q, Scalar(255, 255, 0));
            }
        }
    }
}

int main(int argc, char** argv)
{
    
    
    
    cv::imshow("foo", img_bgr);
    cv::waitKey();
    
    
    return 0;
}
