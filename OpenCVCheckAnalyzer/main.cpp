// Example showing how to read and write images
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv/cvaux.hpp"

#include <iostream>

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
    static int eps = 50;
    
    int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3];
    int x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];
    if (float d = ((float)(x1-x2) * (y3-y4)) - ((y1-y2) * (x3-x4)))
    {
        cv::Point2f pt;
        pt.x = ((x1*y2 - y1*x2) * (x3-x4) - (x1-x2) * (x3*y4 - y3*x4)) / d;
        pt.y = ((x1*y2 - y1*x2) * (y3-y4) - (y1-y2) * (x3*y4 - y3*x4)) / d;
        //-10 is a threshold, the POI can be off by at most 10 pixels
        if(pt.x<min(x1,x2)-eps||pt.x>max(x1,x2)+eps||pt.y<min(y1,y2)-eps||pt.y>max(y1,y2)+eps){
            return Point2f(-1,-1);
        }
        if(pt.x<min(x3,x4)-eps||pt.x>max(x3,x4)+eps||pt.y<min(y3,y4)-eps||pt.y>max(y3,y4)+eps){
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

bool comparator_x(Point2f a,Point2f b){
    return a.x<b.x;
}

bool comparator_y(Point2f a,Point2f b){
    return a.y<b.y;
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
    sort(top.begin(),top.end(),comparator_x);
    sort(bot.begin(),bot.end(),comparator_x);
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


cv::Point2f RotatePoint(const cv::Point2f& p, float rad)
{
    const float x = std::cos(rad) * p.x - std::sin(rad) * p.y;
    const float y = std::sin(rad) * p.x + std::cos(rad) * p.y;
    
    const cv::Point2f rot_p(x, y);
    return rot_p;
}

cv::Point2f RotatePoint(const cv::Point2f& cen_pt, const cv::Point2f& p, float rad)
{
    const cv::Point2f trans_pt = p - cen_pt;
    const cv::Point2f rot_pt   = RotatePoint(trans_pt, rad);
    const cv::Point2f fin_pt   = rot_pt + cen_pt;
    
    return fin_pt;
}


void FindContours()
{
    Mat rgb = imread("check_image.jpg");
    Mat gray;
    cvtColor(rgb, gray, CV_BGR2GRAY);
    
    Mat grad;
    Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(gray, grad, MORPH_GRADIENT, morphKernel);
    
    Mat bw;
    threshold(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
    
    // connect horizontally oriented regions
    Mat connected;
    morphKernel = getStructuringElement(MORPH_RECT, Size(9, 1));
    morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);
    
    // find contours
    Mat mask = Mat::zeros(bw.size(), CV_8UC1);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(connected, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    
    vector<Rect> mrz;
    double r = 0;
    // filter contours
    for(int idx = 0; idx >= 0; idx = hierarchy[idx][0])
    {
        Rect rect = boundingRect(contours[idx]);
        r = rect.height ? (double)(rect.width/rect.height) : 0;
        if ((rect.width > connected.cols * .7) && /* filter from rect width */
            (r > 25) && /* filter from width:hight ratio */
            (r < 36) /* filter from width:hight ratio */
            )
        {
            mrz.push_back(rect);
            rectangle(rgb, rect, Scalar(0, 255, 0), 1);
        }
        else
        {
            rectangle(rgb, rect, Scalar(0, 0, 255), 1);
        }
    }
    if (2 == mrz.size())
    {
        // just assume we have found the two data strips in MRZ and combine them
        
        CvRect rect0 = mrz[0];
        CvRect rect1 = mrz[1];
        
        CvRect max = cvMaxRect(&rect0, &rect1);
        rectangle(rgb, max, Scalar(255, 0, 0), 2);  // draw the MRZ
        
        vector<Point2f> mrzSrc;
        vector<Point2f> mrzDst;
        
        // MRZ region in our image
        mrzDst.push_back(Point2f((float)max.x, (float)max.y));
        mrzDst.push_back(Point2f((float)(max.x+max.width), (float)max.y));
        mrzDst.push_back(Point2f((float)(max.x+max.width), (float)(max.y+max.height)));
        mrzDst.push_back(Point2f((float)max.x, (float)(max.y+max.height)));
        
        // MRZ in our template
        mrzSrc.push_back(Point2f(0.23f, 9.3f));
        mrzSrc.push_back(Point2f(18.0f, 9.3f));
        mrzSrc.push_back(Point2f(18.0f, 10.9f));
        mrzSrc.push_back(Point2f(0.23f, 10.9f));
        
        // find the transformation
        Mat t = getPerspectiveTransform(mrzSrc, mrzDst);
        
        // photo region in our template
        vector<Point2f> photoSrc;
        photoSrc.push_back(Point2f(0.0f, 0.0f));
        photoSrc.push_back(Point2f(5.66f, 0.0f));
        photoSrc.push_back(Point2f(5.66f, 7.16f));
        photoSrc.push_back(Point2f(0.0f, 7.16f));
        
        // surname region in our template
        vector<Point2f> surnameSrc;
        surnameSrc.push_back(Point2f(6.4f, 0.7f));
        surnameSrc.push_back(Point2f(8.96f, 0.7f));
        surnameSrc.push_back(Point2f(8.96f, 1.2f));
        surnameSrc.push_back(Point2f(6.4f, 1.2f));
        
        vector<Point2f> photoDst(4);
        vector<Point2f> surnameDst(4);
        
        // map the regions from our template to image
        perspectiveTransform(photoSrc, photoDst, t);
        perspectiveTransform(surnameSrc, surnameDst, t);
        // draw the mapped regions
        for (int i = 0; i < 4; i++)
        {
            line(rgb, photoDst[i], photoDst[(i+1)%4], Scalar(0,128,255), 2);
        }
        for (int i = 0; i < 4; i++)
        {
            line(rgb, surnameDst[i], surnameDst[(i+1)%4], Scalar(0,128,255), 2);
        }
    }
    
    cv::imshow("rgb", rgb);
    cv::waitKey();
}

int main(void)
{
    using namespace cv;
    
    Mat origImg;
    origImg = imread("check_image.jpg");
    
    int pass = 0;
    vector<vector<cv::Point2f>> corners;

    vector<cv::Point2f> rectangleBounds;
    Mat rot_mat;
    float angle = 0;
    cv::Point2f left, right;
    
    Mat img;
    resize(origImg, img, Size(), 0.25, 0.25, INTER_CUBIC);
    
    while(pass < 2)
    {
        
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        
        
        Mat blur;
        GaussianBlur(gray, blur, Size(3, 3), 0);
        
        Mat edges;
        Canny(blur, edges, 0, 255);
        
        Mat probLineImage;
        cvtColor(edges, probLineImage, CV_GRAY2BGR);
        
        vector<Vec4i> probLines;
        HoughLinesP(edges, probLines, 1, CV_PI/180, 100, 200, 100 );
        
        for( size_t i = 0; i < probLines.size(); i++ )
        {
            Vec4i l = probLines[i];
            line( probLineImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(i * 10,i * 20,255), 3, CV_AA);
        }
        
        corners.clear();
        
        FindConnectedComponents(probLines, edges, corners);
        
        for(int i=0;i<corners.size();i++)
        {
            cv::Point2f center(0,0);
            
            if(corners[i].size()<4)
            {
                corners.erase(corners.begin() + i);
                i--;
                continue;
            }
            
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
                circle(probLineImage, point, 50, Scalar(0, 0, 255), 10);
            }
        }
        
        rectangleBounds = corners[0];
        
        sort(rectangleBounds.begin(), rectangleBounds.end(), comparator_y);
        
        // Last two elements are the bottom
        
        if (rectangleBounds[2].x < rectangleBounds[3].x)
        {
            left = rectangleBounds[2];
            right = rectangleBounds[3];
            
            // In this case, right is lower down
            
            float opposite = right.y - left.y;
            float adjacent = right.x - left.x;
            angle = cvFastArctan(opposite, adjacent);
        }
        else
        {
            right = rectangleBounds[2];
            left = rectangleBounds[3];
            
            // In this case, left is lower down
            
            float opposite = left.y - right.y;
            float adjacent = right.x - left.x;
            angle = cvFastArctan(opposite, adjacent);
            
            angle *= -1;
            
        }
        
        rot_mat = getRotationMatrix2D(left, angle, 1.0);
        Mat dst;
        warpAffine(img, dst, rot_mat, img.size());
        
        Mat dstOrig;
        warpAffine(origImg, dstOrig, rot_mat, origImg.size());

        /*cv::imshow("blur", blur);
        cv::imshow("x", edges);
        cv::imshow("probLines", probLineImage);
        cv::imshow("rotated", dst);
        cv::waitKey();*/
        
        dst.copyTo(img);
        dstOrig.copyTo(origImg);
        
        //cv::destroyAllWindows();
        
        pass++;
    }
    
    
    vector<cv::Point2f> rotatedPoints;
    
    for (auto point : corners[0])
    {
        cv::Point2f rotatedPoint = RotatePoint(left, point, (-angle / 180.0f) * CV_PI);
        rotatedPoint.x *= 4;
        rotatedPoint.y *= 4;
        
        rotatedPoints.push_back(rotatedPoint);
    }
    
    
    Rect r = boundingRect(rotatedPoints);
    
    // Crop out everything except bottom numbers
    r.y += r.height * 0.85;//r.height * 0.9;
    r.height *= 0.12;

    Mat finalImage;
    origImg(r).copyTo(finalImage);
    
    cv::Mat img2gray;
    cv::cvtColor(finalImage, img2gray, cv::COLOR_BGR2GRAY);
    
    cv::Mat inverted;
    cv::bitwise_not(img2gray, inverted);
    
    cv::Mat mask;
    cv::threshold(img2gray, mask, 100, 255, cv::THRESH_BINARY);
    
    cv::imwrite("extracted_check_no_p.jpg", mask);
    

/*
    // Define the destination image
    cv::Mat quad = cv::Mat::zeros(r.height, r.width, CV_8UC3);
    // Corners of the destination image
    std::vector<cv::Point2f> quad_pts;
    quad_pts.push_back(cv::Point2f(0, 0));
    quad_pts.push_back(cv::Point2f(quad.cols, 0));
    quad_pts.push_back(cv::Point2f(quad.cols, quad.rows));
    quad_pts.push_back(cv::Point2f(0, quad.rows));
    // Get transformation matrix
    cv::Mat transmtx = cv::getPerspectiveTransform(rotatedPoints, quad_pts);
    // Apply perspective transformation
    cv::warpPerspective(origImg, quad, transmtx, quad.size());
    std::stringstream ss;
    ss<<0<<".jpg";
    imshow(ss.str(), quad);

    
    cv::imshow("Final", img);
    cv::waitKey();
    
    cv::imwrite("extracted_check.jpg", quad);*/
    
    return 0;
}
