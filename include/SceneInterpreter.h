//
// Created by jbs on 21. 10. 31..
//

// todo yaml parameterize

#ifndef ZED_OPEN3D_SCENEINTERPRETER_H
#define ZED_OPEN3D_SCENEINTERPRETER_H


#include <ZedUtils.h>
#include <Open3dUtils.h>
#include <Misc.h>
#include <mutex>
#include "opencv2/cudaarithm.hpp"


using namespace std;

static bool activeWhile = true;


namespace iswy { // i see with you
    struct ElapseMonitor {
        static map<string,float> monitorResult;
        float elapseMeasured =-1 ;
        string tag;
        misc::Timer* timerPtr;

        ElapseMonitor(string tag): tag(tag) {
            timerPtr = new misc::Timer();
        }
        ~ElapseMonitor();
    };


    void INThandler(int sig);

    struct SceneParam {


    };
    // visualization with open3d and opencv ...
    struct VisParam{
        string nameImageWindow = "image perception";
        float objectPixelAlpha = 0.5;

        int const attentionColors[4][3] = {{51,51,255}, {0,128,255}, {0,255,255},{51,255,51}}; // BGR

    };


    struct CameraParam;

    struct ZedState{ // raw camera state

        sl::Pose pose;
        sl::Mat image;
        sl::Mat depth;
        sl::Objects humans;
        sl::ObjectData actor;

        sl::Camera camera;


        bool grab(const CameraParam & runParam);
        bool markHumanPixels (cv::cuda::GpuMat& maskMat);
        Eigen::Matrix4f getPoseMat() const;
    };

    struct CameraParam{
    private:
        bool isSvo = true;
        // these will be filled once camera opened !
        float fx;
        float fy;
        float cx;
        float cy;
        int width;
        int height;
        sl::InitParameters* initParameters;
        sl::ObjectDetectionParameters* detectionParameters;
        sl::RuntimeParameters* runtimeParameters;
        sl::ObjectDetectionRuntimeParameters* objectDetectionRuntimeParameters;
        sl::PositionalTrackingParameters* positionalTrackingParameters;

    public:
        CameraParam(string paramterFilePath,string svoFileDir);
        bool open(ZedState& zed);
        sl::RuntimeParameters getRtParam() const {return *runtimeParameters;};
        sl::ObjectDetectionRuntimeParameters getObjRtParam() const {return *objectDetectionRuntimeParameters;}
        sl::PositionalTrackingParameters getPoseParam() const {return *positionalTrackingParameters;}
        cv::Size getCvSize() const {return cv::Size(width,height);};
        cv::Point2f project(const Eigen::Vector3f& pnt) const{
            return {fx*pnt.x()/pnt.z() + cx, fx*pnt.y()/pnt.z() + cy};};
        inline void unProject (cv::Point uv, float depth, float& xOut, float& yOut, float & zOut) const{
            zOut = depth;
            xOut = (uv.x - cx) * depth / fx;
            yOut = (uv.y - cy) * depth / fy;
        };
        Eigen::Matrix3d getCameraMatrix() const ;
        open3d::core::Tensor getO3dIntrinsicTensor(
                open3d::core::Device dType = open3d::core::Device("CUDA:0")) const;

    };


    struct AttentionParam{
        int ooi = 39; // bottle
        float gazeFov = M_PI /3.0 *2.0; // cone (height/rad) = tan(gazeFov/2)
        float gazeImportance = 0.4;
    };


    struct ObjectDetectParam{
        string rootDir = "/home/jbs/lib/see_with_you/param/";
        string modelConfig = rootDir + "yolov4.cfg";
        string modelWeight = rootDir + "yolov4.weights";
        string classNameDir = rootDir + "coco.names";
        vector<string> classNames;
        float confidence = 0.2;
        float nmsThreshold = 0.2;
        float objectDepthMin = 0.01;
        float objectDepthMax = 5.0;
        float objectDimensionAlongOptical = 1.0;


    };

    struct humanDetectParam{

    };

    struct DetectedObject{
        cv::Rect boundingBox;
//        cv::cuda::GpuMat mask; // 255 masking in the bounding box
        cv::Mat mask_cpu; // 255 masking in the bounding box
        int classLabel;
        string className;
        float confidence ;
        Eigen::Vector3f centerPoint;

        void drawMe (cv::Mat& image,float alpha ,int ooi) const;
        void findMe (const cv::cuda::GpuMat& depth,
                     ObjectDetectParam param, CameraParam camParam );

        float attentionCost = INFINITY;
        void updateAttentionCost (float cost) {attentionCost = cost; }

    };

    struct AttentionEvaluator{
        zed_utils::Gaze gaze;
        Eigen::Vector3f leftHand; // left hand location
        Eigen::Vector3f rightHand; // right hand location
        float evalAttentionCost(DetectedObject& object, AttentionParam param, bool updateObject = true);
        bool isValid() {return (! isnan(leftHand.norm()) && (! isnan(leftHand.norm())) && gaze.isValid()); }
        void drawMe (cv::Mat& image, CameraParam camParam); // todo extrinsic
    };

    struct VisOpenCv{
        cv::Mat image;
        vector<DetectedObject> curObjVis; // renewed from update thread


    };

    struct DeviceData{
        // raw image bound with ZED
        cv::cuda::GpuMat imageCv;
        cv::cuda::GpuMat imageCv3ch;
        cv::cuda::GpuMat depthCv;
        cv::cuda::GpuMat fgMask; // {human, objects} = 255
        cv::cuda::GpuMat bgDepthCv; // depth - {human, objects}
    };


    class SceneInterpreter {
    private:
        bool isGrab = false;
        bool imshowWindowOpened = false;
        mutex mutex_; // mutex between cameraThread and vis thread
        // visualizers
        VisOpenCv visOpenCv;
        VisParam paramVis;

        // detector
        ObjectDetectParam paramDetect;
        cv::dnn::Net net;
        vector<DetectedObject> detectedObjects;
        void detect();

        // zed
        CameraParam zedParam;
        ZedState zedState;

        // attention
        AttentionParam paramAttention;
        AttentionEvaluator attention;
        void drawAttentionScores(cv::Mat& image, const vector<DetectedObject>& objs) const;

        // variables in device
        DeviceData deviceData;

        void bindDevice();
        void forwardToVisThread();
        void visualize();



    public:
        SceneInterpreter();
        void cameraThread();
        void visThread();
        ~SceneInterpreter() {};
    };

}
#endif //ZED_OPEN3D_SCENEINTERPRETER_H


