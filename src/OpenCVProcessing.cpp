/*
 * Bebop2Demo.cpp
 *
 *  Created on: Feb 1, 2019
 *      Author: slascos
 */
#include "OpenCVProcessing.h"

using namespace std;
using namespace cv;

#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
//#include "opencv2/core/cuda.hpp"
//#include "opencv2/cudaimgproc.hpp"

using namespace cv;
using namespace cv::dnn;
using namespace cv::xfeatures2d;

void harrisCorner(Mat &grayImage, Mat &outputImage)
{
    Mat corners, cornersNorm, cornersNormScaled;
    int thresh = 100;

    cornerHarris(grayImage, corners, 7, 5, 0.05, BORDER_DEFAULT);
    normalize(corners, cornersNorm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(cornersNorm, cornersNormScaled);

    // Draw circles around corners
    for (int j = 0; j < cornersNorm.rows; j++) {
        for (int i = 0; i < cornersNorm.cols; i++) {
            if ( (int) cornersNorm.at<float>(j,i) > thresh ) {
                circle (cornersNormScaled, Point(i,j), 5, Scalar(255), 2, 8, 0);
            }
        }
    }
    cornersNormScaled.copyTo(outputImage);
}


void targetDetectionSetup(shared_ptr<Mat> refObjectPtr)
{
    *refObjectPtr = imread("refObjectSmall.png", IMREAD_GRAYSCALE);
    if ((*refObjectPtr).empty()) {
        cout << "Failed to open refObjectSmall.png" << endl;
    }
}
void targetDetection(shared_ptr<Mat> imgObjectPtr, shared_ptr<Mat> imageScenePtr)
{
    Mat img_object = (*imgObjectPtr);
    Mat img_scene = (*imageScenePtr);

    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create( minHessian );
    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;
    detector->detectAndCompute( img_object, noArray(), keypoints_object, descriptors_object );
    detector->detectAndCompute( img_scene, noArray(), keypoints_scene, descriptors_scene );

    if (keypoints_object.empty() || keypoints_scene.empty()) {
        return; // no keypoints found
    } else {
        //cout << "Found keypoints: " << keypoints_object.size() << "/" << keypoints_scene.size() << endl;
    }
    if (descriptors_object.empty() || descriptors_scene.empty()) {
        return; // no descriptors found
    } else {
        //cout << "Found descriptors: " << descriptors_object.size() << "/" << descriptors_scene.size() << endl;
    }

    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors_object, descriptors_scene, knn_matches, 2 );


    //-- Filter matches using the Lowe's ratio test
    //cout << "Found matches: " << knn_matches.size() << endl;
    const float ratio_thresh = 0.75f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    //-- Draw matches
    Mat img_matches;
    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    if (good_matches.size() < 1) { return; }

    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }
    Mat H = findHomography( obj, scene, RANSAC );

    if (H.empty()) { return; }
    //else { cout << "Found homography: " << H << endl; }

    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f( (float)img_object.cols, 0 );
    obj_corners[2] = Point2f( (float)img_object.cols, (float)img_object.rows );
    obj_corners[3] = Point2f( 0, (float)img_object.rows );
    std::vector<Point2f> scene_corners(4);
    perspectiveTransform( obj_corners, scene_corners, H);

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_matches, scene_corners[0] + Point2f((float)img_object.cols, 0),
          scene_corners[1] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4 );
    line( img_matches, scene_corners[1] + Point2f((float)img_object.cols, 0),
          scene_corners[2] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[2] + Point2f((float)img_object.cols, 0),
          scene_corners[3] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[3] + Point2f((float)img_object.cols, 0),
          scene_corners[0] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );

    //-- Show detected matches
    img_scene = img_matches;
    img_scene.copyTo(*imageScenePtr);
}

void openCVProcessing(shared_ptr<Mat> imageToProcess, bool *processingDone)
{
    // Convert to grayscale
    Mat grayImage;
    cv::cvtColor(*imageToProcess, grayImage, COLOR_BGR2GRAY);

    Mat outputImage;
    harrisCorner(grayImage, outputImage);
    outputImage.copyTo(*imageToProcess);

    // Target Detection
//    shared_ptr<Mat> refObjectPtr = make_shared<Mat>();
//    shared_ptr<Mat> sceneImagePtr = make_shared<Mat>();
//    *sceneImagePtr = grayImage; // input scene image
//    targetDetectionSetup(refObjectPtr);
//    targetDetection(refObjectPtr, sceneImagePtr);
//    (*sceneImagePtr).copyTo(*imageToProcess);

    *processingDone = true;
}
