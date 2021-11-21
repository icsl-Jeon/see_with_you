//
// Created by junbs on 11/21/2021.
//

#include <ZedUtils.h>
#include <Misc.h>
#include <SceneInterpreter.h>
#include "opencv2/cudaarithm.hpp"
#include "opencv2/rgbd.hpp"

using namespace iswy;

int main(){

    // Initialize ZED
    string svoFileDir = "C:\\Users\\junbs\\OneDrive\\Documents\\ZED\\HD1080_SN28007858_14-25-48.svo";
    CameraParam zedParam(" ",svoFileDir);
    ZedState zedState;
    zedParam.open(zedState);

    if (zedState.camera.isOpened()){
        std::cout << "Camera opened !" << std::endl;

    }else{
        std::cerr << "Camera not opened" << std::endl;
        return 0;
    }

    // Opencv image
    cv::cuda::GpuMat deviceImage,deviceDepth,deviceImage3ch;
    deviceImage = zed_utils::slMat2cvMatGPU(zedState.image);
    deviceDepth = zed_utils::slMat2cvMatGPU(zedState.depth);

    cv::Mat hostImage;
    cv::Mat hostDepth;
    cv::namedWindow("Image", cv::WINDOW_KEEPRATIO);
    cv::resizeWindow("Image",1920,1080);

    // Opencv Tsdf
//    cv::setUseOptimized( true );
    cv::Ptr<cv::kinfu::Params> params;
    params = cv::kinfu::Params::defaultParams();
    params->frameSize = zedParam.getCvSize();

    Eigen::Matrix3f intrinsic = zedParam.getCameraMatrix();
    for (int r = 0 ; r < 3 ; r++)
        for (int c = 0 ; c < 3 ; c++)
            params->intr(r,c) = intrinsic(r,c);
    params->depthFactor = 1.0;

    cv::Ptr<cv::kinfu::KinFu> kinfu
        = cv::kinfu::KinFu::create(params);


    cv::UMat uImage;
    cv::UMat uDepth;     // Camera thread

    while (true){
        // image
        zedState.grab(zedParam);
        cv::cuda::cvtColor(deviceImage, deviceImage3ch, cv::COLOR_BGRA2BGR);

        deviceImage3ch.download(hostImage);
        deviceDepth.download(hostDepth);

        // for opencl acceleration...
        hostImage.copyTo(uImage);
        hostDepth.copyTo(uDepth);


        cv::imshow("Depth",uDepth);
        cv::waitKey(1);

        // tsdf
        {
            ElapseMonitor monitor("Kinfu Update");

            if ( ! kinfu->update(uDepth))
                kinfu->reset();
        }

    }

    return 0;
}