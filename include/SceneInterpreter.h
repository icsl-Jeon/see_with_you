/**
 * SceneInterpret receives the camera sensing data and reconstruct the following information
 * 1. OOIs (meshes + attention score)
 * 2. Environment spatial mapping (currently mesh)
 */

#ifndef ZED_OPEN3D_SCENEINTERPRETER_H
#define ZED_OPEN3D_SCENEINTERPRETER_H


#include <ZedUtils.h>
#include <Open3dUtils.h>
#include <Misc.h>
#include <mutex>

//////////////////////////////////////////////
using namespace std;
namespace o3d_core = open3d::core;
namespace o3d_tensor =  open3d::t::geometry ;
//////////////////////////////////////////////

namespace iswy { // i see with you

    ///////////////////////////////////////////////////////////
    //                     Parameters                        //
    //////////////////////////////////////////////////////////

    namespace param {
        // visualization with open3d and opencv ...
        struct VisParam{
            string nameImageWindow = "image perception";
            float objectPixelAlpha = 0.5;
            int const attentionColors[4][3] = {{51,51,255}, {0,128,255}, {0,255,255},{51,255,51}}; // BGR
        };

        struct AttentionParam{
            int ooi = 39; // bottle
            float gazeFov = M_PI /3.0 *2.0; // cone (height/rad) = tan(gazeFov/2)
            float gazeImportance = 0.4;
        };

        struct ObjectDetectParam{
            bool isActive = true;

            string modelConfig;
            string modelWeight;
            string classNameDir;
            vector<string> classNames;

            float confidence = 0.2;
            float nmsThreshold = 0.2;
            float objectDepthMin = 0.01;
            float objectDepthMax = 5.0;
            float objectDimensionAlongOptical = 1.0;
            bool parseFrom(string configFile);
        };

        struct MappingParam{
            float tsdfVoxelSize = 0.015;
            float tsdfTrunc = 0.04;
            float tsdfBlockCount = 1000;
        };
    }

    ///////////////////////////////////////////////////////////
    //               Managing sensor data                    //
    //////////////////////////////////////////////////////////

    /// \brief Shared resource for extracting sensing information (deprecated)
    struct DeviceData{

        DeviceData() = default;

        // data type for open3d
        o3d_core::Device device_gpu = o3d_core::Device("CUDA:0");
        o3d_core::Device device_cpu = o3d_core::Device("CPU:0");
        o3d_core::Dtype rgbType = o3d_core::Dtype::UInt8;
        o3d_core::Dtype depthType = o3d_core::Dtype::Float32;

        // opencv image bound with ZED
        cv::cuda::GpuMat imageCv;
        cv::cuda::GpuMat imageCv3ch;
        cv::cuda::GpuMat depthCv;
        cv::cuda::GpuMat fgMask; // {human, objects} = 255
        cv::cuda::GpuMat bgDepthCv; // depth - {human, objects}

        // open3d image bound with opencv image
        shared_ptr<o3d_core::Blob> rgbBlob;
        shared_ptr<o3d_core::Blob> depthBlob;
        o3d_core::Tensor rgbTensor ;
        o3d_core::Tensor depthTensor ;
        o3d_tensor::Image imageO3d;
        o3d_tensor::Image depthO3d;

        // open3d volumetric TSDF
        o3d_tensor::VoxelBlockGrid* volumePtr; // TODO: new open3d master doesn' have TSDFVoxelGrid.h
    };

    struct HostData{
        o3d_core::Device device_cpu = o3d_core::Device("CPU:0");
        o3d_core::Dtype rgbType = o3d_core::Dtype::UInt8;
        o3d_core::Dtype depthType = o3d_core::Dtype::Float32;

        cv::Mat imageCv3ch;
        cv::Mat depthCv;
        cv::Mat fgMask; // {human, objects} = 255
        cv::Mat imageCv;
        cv::Mat bgDepthCv; // depth - {human, objects}

        shared_ptr<o3d_core::Blob> rgbBlob;
        shared_ptr<o3d_core::Blob> depthBlob;
        o3d_core::Tensor rgbTensor ;
        o3d_core::Tensor depthTensor ;
        o3d_tensor::Image imageO3d;
        o3d_tensor::Image depthO3d;

    };

    /// \brief  Raw camera sensing data and its data in gpu
    struct Camera{
    private:
        bool isBound = false;
        bool isGrab = false;
        DeviceData deviceData; /// Raw data in gpu
        HostData hostData; /// Raw data in cpu
        // Core sensing data. They should be bound each other for shallow copy
        zed_utils::CameraParam zedParam;
        zed_utils::ZedState zedState; /// sensor raw data
        void bindDevice(); /// binding between sl objects - cv objects - open3d objects in gpu memory
        void bindHost();  /// binding between sl objects - cv objects - open3d objects in cpu memory

    public:
        Camera(string configFile); /// open camera with config file
        open3d::geometry::Image getImageO3d() const;
        cv::Mat getImageCv() const;
        sl::ObjectData getHumanZed() const;

        bool grab(); /// update device data with incoming sensor

    };

    /// \brief Detect various objects in image and find their pixels and locations (not actor)
    struct ObjectDetector{
        param::ObjectDetectParam paramDetect;
        cv::dnn::Net net;
        bool detect(const cv::Mat& imageCv3ch);
        ObjectDetector(string configFile);
    };


    ///////////////////////////////////////////////////////////
    //           objects extracted from sensing              //
    //////////////////////////////////////////////////////////


    struct DetectedObject{
        cv::Rect boundingBox; /// assuming the image size does not change
//        cv::cuda::GpuMat mask; /// 255 masking in the bounding box
        cv::Mat mask_cpu; /// 255 masking in the bounding box
        int classLabel; /// follows yolo labeling  (cfg)
        string className;
        float confidence;
        Eigen::Vector3f centerPoint; /// w.r.t world coordinate
        float attentionCost = INFINITY; /// degree of interest from actor

        /// \brief Overlay the mask for object with coloring mapped to attention cost
        /// \param image target image for overlay
        /// \param alpha transparency.
        /// \param ooi 
        void drawMe (cv::Mat& image,float alpha ,int ooi) const;

        /// \brief finding position of the object based on the depth image and bounding box
        /// \param depth
        /// \param param
        /// \param camParam
        void findPosFromDepth (const cv::cuda::GpuMat& depth,
                               param::ObjectDetectParam param, zed_utils::CameraParam camParam );
    };


    class Gaze{
    private:
        Eigen::Vector3f root;
        Eigen::Vector3f direction; // to be normalized
        Eigen::Matrix4f transformation; // z: normal outward from face, x: left eye to right eye (w.r.t actor)

    public:
        Gaze() = default;
        Gaze(const sl::ObjectData& humanObject);
        Eigen::Matrix4f getTransformation() const;
        bool isValid() const ;
        float measureAngleToPoint(const Eigen::Vector3f & point) const;
        tuple<Eigen::Vector3f,Eigen::Vector3f> getGazeLineSeg(float length)  const; // p1 ~ p2

    };

    struct Actor{
        Gaze gaze;
        Eigen::Vector3f leftHand; // left hand location
        Eigen::Vector3f rightHand; // right hand location
        float evalAttentionCost(DetectedObject& object, param::AttentionParam param, bool updateObject = true);
        bool isValid() {return (! isnan(leftHand.norm()) && (! isnan(leftHand.norm())) && gaze.isValid()); }
        void drawMe (cv::Mat& image, zed_utils::CameraParam camParam); // todo extrinsic
    };


    ///////////////////////////////////////////////////////////
    //                      Visualizer                      //
    //////////////////////////////////////////////////////////

    /// \brief visualize the belows:
    /// 1. Detection of objects with score coloring
    /// 2. Actor tracking and its skeleton
    struct Visualizer{
        param::VisParam param_;


        void drawAttentionScores(cv::Mat& image, const vector<DetectedObject>& objs) const;

    };


    ///////////////////////////////////////////////////////////
    //                 Integrated processor                  //
    //////////////////////////////////////////////////////////

    /// \brief Process the incoming data for framing evaluation.
    /// 1. performs rgb + depth camera sensing
    /// 2. detect objects + actor. Gaze of actor is also estimated.
    /// 3. Each object is assigned a attention score based on gaze, hands, context
    /// 4. Volumetric mapping is performed by TSDF
    class SceneInterpreter {
    private:
        // Sensing
        Camera camera; /// Receives RGB + Depth and perform spatial mapping and actor tracking
        ObjectDetector detector; /// Extract OOIs (not actor)

        // Extracted foreground objects
        vector<DetectedObject> detectedObjects;
        Actor detectedActor;

        // Misc pipeline
        mutex mutex_; // mutex between perceptionThread and vis thread
        Visualizer visualizer;

        void forwardToVisThread();
        void visualize();

    public:
        /// \brief This handles the incoming data for framing evaluation.
        /// 1. performs rgb + depth camera sensing
        /// 2. detect objects + actor. Gaze of actor is also estimated.
        /// 3. Each object is assigned a attention score based on gaze, hands, context
        /// 4. Volumetric mapping is performed by TSDF
        SceneInterpreter(string configFile);

        /// \brief Run loop for sensing data
        void perceptionThread();

        /// \brief run visualization by open3d
        void visualizationThread();
        ~SceneInterpreter() {};

        /// \brief update perception
        bool grab();


        /// \brief get the latest image in open3d legacy format
        open3d::geometry::Image getImageO3d() const;
        open3d::geometry::LineSet getSkeletonO3d() const;

        void disableObjectDetection();
        void enableObjectDetection();

    };

}
#endif //ZED_OPEN3D_SCENEINTERPRETER_H


