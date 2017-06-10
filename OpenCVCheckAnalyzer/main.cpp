// Example showing how to read and write images
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv/cvaux.hpp"

int main(int argc, char** argv)
{
    
    // Load an image from file - change this based on your image name
    cv::Mat img = cv::imread("apple-16.jpg", CV_LOAD_IMAGE_UNCHANGED);
    if(img.data == NULL)
    {
        fprintf(stderr, "failed to load input image\n");
        return -1;
    }
    
    cv::Mat subImg = img(cv::Range(450, 550), cv::Range(450, 550));
    
    subImg.copyTo(img(cv::Range(0,100), cv::Range(0,100)));

    // Write the image to a file with a different name,
    // using a different image format -- .png instead of .jpg
    if(!cv::imwrite("test_file.png", img))
    {
        fprintf(stderr, "failed to write image file\n");
    }
    
    return 0;
}
