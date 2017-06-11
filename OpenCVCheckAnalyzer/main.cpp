// Example showing how to read and write images
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv/cvaux.hpp"

using namespace cv;

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
        Canny(frame, edges, 0, 200);
        
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

void ExerciseTwelve()
{
    using namespace cv;
    
    Mat img;
    img = imread("opencv-python-foreground-extraction-tutorial.jpg");
    //printf("%d, %d, %d, %d", img.type(), img.depth(), img.channels());
    
    Mat result;
    Mat bgdModel;
    Mat fgdModel;
    
    Rect rect(50, 50, 300, 200);
    
    grabCut(img, result, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT);
    
    compare(result, GC_PR_FGD, result, CMP_EQ);
    
    Mat foreground(img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
    img.copyTo(foreground, result);
    
    cv::imshow("foo", foreground);
    cv::waitKey();
}

cv::Point2f computeIntersect(cv::Vec4i a, cv::Vec4i b)
{
    int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3];
    int x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];
    if (float d = ((float)(x1-x2) * (y3-y4)) - ((y1-y2) * (x3-x4)))
    {
        cv::Point2f pt;
        pt.x = ((x1*y2 - y1*x2) * (x3-x4) - (x1-x2) * (x3*y4 - y3*x4)) / d;
        pt.y = ((x1*y2 - y1*x2) * (y3-y4) - (y1-y2) * (x3*y4 - y3*x4)) / d;
        //-10 is a threshold, the POI can be off by at most 10 pixels
        if(pt.x<min(x1,x2)-10||pt.x>max(x1,x2)+10||pt.y<min(y1,y2)-10||pt.y>max(y1,y2)+10){
            return Point2f(-1,-1);
        }
        if(pt.x<min(x3,x4)-10||pt.x>max(x3,x4)+10||pt.y<min(y3,y4)-10||pt.y>max(y3,y4)+10){
            return Point2f(-1,-1);
        }
        return pt;
    }
    else
        return cv::Point2f(-1, -1);
}

void FindConnectedComponents(vector<Vec4i>& lines, Mat& img2, vector<vector<cv::Point2f>>& corners)
{
    int* poly = new int[lines.size()];
    
    for(int i=0;i<lines.size();i++)poly[i] = - 1;
    
    int curPoly = 0;
    
    for (int i = 0; i < lines.size(); i++)
    {
        for (int j = i+1; j < lines.size(); j++)
        {
            cv::Point2f pt = computeIntersect(lines[i], lines[j]);
            
            if (pt.x >= 0 && pt.y >= 0&&pt.x<img2.size().width&&pt.y<img2.size().height){
                
                if(poly[i]==-1&&poly[j] == -1){
                    vector<Point2f> v;
                    v.push_back(pt);
                    corners.push_back(v);
                    poly[i] = curPoly;
                    poly[j] = curPoly;
                    curPoly++;
                    continue;
                }
                if(poly[i]==-1&&poly[j]>=0){
                    corners[poly[j]].push_back(pt);
                    poly[i] = poly[j];
                    continue;
                }
                if(poly[i]>=0&&poly[j]==-1){
                    corners[poly[i]].push_back(pt);
                    poly[j] = poly[i];
                    continue;
                }
                if(poly[i]>=0&&poly[j]>=0){
                    if(poly[i]==poly[j]){
                        corners[poly[i]].push_back(pt);
                        continue;
                    }
                    
                    for(int k=0;k<corners[poly[j]].size();k++){
                        corners[poly[i]].push_back(corners[poly[j]][k]);
                    }
                    
                    corners[poly[j]].clear();
                    poly[j] = poly[i];
                    continue;
                }
            }
        }
    }
    
    printf("Foo");
}

bool comparator(Point2f a,Point2f b){
    return a.x<b.x;
}

void sortCorners(std::vector<cv::Point2f>& corners, cv::Point2f center)
{
    std::vector<cv::Point2f> top, bot;
    for (int i = 0; i < corners.size(); i++)
    {
        if (corners[i].y < center.y)
            top.push_back(corners[i]);
        else
            bot.push_back(corners[i]);
    }
    sort(top.begin(),top.end(),comparator);
    sort(bot.begin(),bot.end(),comparator);
    cv::Point2f tl = top[0];
    cv::Point2f tr = top[top.size()-1];
    cv::Point2f bl = bot[0];
    cv::Point2f br = bot[bot.size()-1];
    corners.clear();
    corners.push_back(tl);
    corners.push_back(tr);
    corners.push_back(br);
    corners.push_back(bl);
}

int main(int argc, char** argv)
{
    using namespace cv;
    
    Mat img;
    img = imread("check_image_small.jpg");
    
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    
    Mat blur;
    GaussianBlur(gray, blur, Size(3, 3), 0);
    
    Mat edges;
    Canny(blur, edges, 0, 255);
    
    vector<Vec2f> lines;
    HoughLines(edges, lines, 1, CV_PI/180, 100, 0);
    
    Mat lineImage;
    cvtColor(edges, lineImage, CV_GRAY2BGR);
    
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line( lineImage, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
    }
    
    Mat probLineImage;
    cvtColor(edges, probLineImage, CV_GRAY2BGR);
    
    vector<Vec4i> probLines;
    HoughLinesP(edges, probLines, 1, CV_PI/180, 100, 200, 100 );
    
    for( size_t i = 0; i < probLines.size(); i++ )
    {
        Vec4i l = probLines[i];
        line( probLineImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(i * 10,i * 20,255), 3, CV_AA);
    }
    
    vector<vector<cv::Point2f>> corners;
    
    FindConnectedComponents(probLines, edges, corners);
    
    for(int i=0;i<corners.size();i++)
    {
        cv::Point2f center(0,0);
        if(corners[i].size()<4)continue;
        for(int j=0;j<corners[i].size();j++){
            center += corners[i][j];
        }
        center *= (1. / corners[i].size());
        sortCorners(corners[i], center);
    }
    
    
    for (auto corner : corners)
    {
        for (auto point : corner)
        {
            circle(probLineImage, point, 5, Scalar(0, 0, 255));
        }
    }
    
    
    cv::imshow("blur", blur);
    cv::imshow("x", edges);
    cv::imshow("lines", lineImage);
    cv::imshow("probLines", probLineImage);
    cv::waitKey();
    
    
    return 0;
}
