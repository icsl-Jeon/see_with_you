#ifndef ZED_OPEN3D_ZEDUTILS_H
#define ZED_OPEN3D_ZEDUTILS_H

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cvconfig.h>
#include "opencv2/cudaimgproc.hpp"
#include <opencv2/cudawarping.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <tuple>
#include <open3d/Open3D.h>
#include <open3d/t/geometry/RaycastingScene.h>
#include <open3d/core/MemoryManager.h>
#include "Misc.h"
#include <yaml-cpp/yaml.h>


using namespace std;

namespace zed_utils {
    using namespace sl;

    bool parseArgs(int argc, char **argv, sl::InitParameters &param);


    cv::Mat slMat2cvMat(sl::Mat &input);

    cv::cuda::GpuMat slMat2cvMatGPU(sl::Mat &input); // currently, not working

    int getOCVtype(sl::MAT_TYPE type);

    void print(string msg_prefix, ERROR_CODE err_code, string msg_suffix);


    struct CameraParam;

    /// \brief Wrapping structure for sl- objects (raw data from ZED SDK)
    struct ZedState{ // raw camera state

        sl::Camera camera;

        sl::Pose pose;
        sl::Mat image;
        sl::Mat depth;
        sl::Objects humans;
        sl::ObjectData actor;

        bool isCameraOpen() {return camera.isOpened();}
        bool grab(const CameraParam & runParam);
        bool markHumanPixels (cv::cuda::GpuMat& maskMat);
        Eigen::Matrix4f getPoseMat() const;
    };

    /// \brief This opens zed sl::Camera with given initialization params (from users) and fills camera info
    /// \details Also, performs simple operation for projection and un-projection
    struct CameraParam{
    private:
        bool isSvo = true;

        // Init params when opening ZED
        sl::InitParameters* initParameters;
        sl::ObjectDetectionParameters* detectionParameters;
        sl::RuntimeParameters* runtimeParameters;
        sl::ObjectDetectionRuntimeParameters* objectDetectionRuntimeParameters;
        sl::PositionalTrackingParameters* positionalTrackingParameters;

        // Will be read once camera opened
        float fx;
        float fy;
        float cx;
        float cy;
        int width;
        int height;

    public:
        /// \brief Initialize starting params for camera from users
        /// \param parameterFilePath yaml file
        CameraParam(string parameterFilePath);

        /// \brief initialize zed sl objects and its data
        /// \param zed
        /// \return
        bool init(ZedState& zed);

        // Simple get- functions
        sl::RuntimeParameters getRtParam() const {return *runtimeParameters;};
        sl::ObjectDetectionRuntimeParameters getObjRtParam() const {return *objectDetectionRuntimeParameters;}
        sl::PositionalTrackingParameters getPoseParam() const {return *positionalTrackingParameters;}
        cv::Size getCvSize() const {return cv::Size(width,height);};
        Eigen::Matrix3d getCameraMatrix() const ;
        open3d::core::Tensor getO3dIntrinsicTensor(
                open3d::core::Device dType = open3d::core::Device("CUDA:0")) const;

        /// \brief project 3D point in world frame to the camera image
        /// \param pnt
        /// \return pixel (u,v) in the image
        cv::Point2f project(const Eigen::Vector3f& pnt) const;

        /// \brief reconstruct 3D point when pixel and depth are given
        /// \param uv
        /// \param depth
        /// \param xOut
        /// \param yOut
        /// \param zOut
        void unProject (cv::Point uv, float depth, float& xOut, float& yOut, float & zOut) const;

    };
}

#endif //ZED_OPEN3D_ZEDUTILS_H
