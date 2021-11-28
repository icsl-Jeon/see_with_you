//
// Created by junbs on 11/21/2021.
//

#include <ZedUtils.h>
#include <Misc.h>
#include <SceneInterpreter.h>
#include "opencv2/cudaarithm.hpp"
#include "opencv2/rgbd.hpp"
#include <Open3dUtils.h>


using namespace iswy;

namespace o3d_tensor =  open3d::t::geometry ;
namespace o3d_legacy = open3d::geometry;
namespace o3d_core = open3d::core;
namespace o3d_vis = open3d::visualization;


int main(){
    // OPENGL
    // Open3D links all opengl stuff (glew, glfw, opengl.... Check Open3D\include\open3d\3rdparty)
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // do not open window
    GLFWwindow* window = glfwCreateWindow(1920,
                                          1080,
                                          "render", NULL, NULL);

    if (window == NULL){
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return 0;
    }
    glfwMakeContextCurrent(window);


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
    deviceImage3ch = cv::cuda::createContinuous(zedParam.getCvSize(),CV_8UC3);
    uint row = zedParam.getCvSize().height;
    uint col = zedParam.getCvSize().width;

    cv::Mat hostImage;
    cv::Mat hostDepth;

    // Open3D image binding

    bool isVisInit = false;

    o3d_core::Device device_gpu("CUDA:0");
    o3d_core::Device device_cpu("CPU:0");
    o3d_core::Dtype rgbType = o3d_core::Dtype::UInt8;
    o3d_core::Dtype depthType = o3d_core::Dtype::Float32;

    auto rgbBlob = std::make_shared<o3d_core::Blob>(
            device_gpu,deviceImage3ch.cudaPtr(), nullptr);
    auto  rgbTensor = o3d_core::Tensor({row,col,3},{3*col,3,1},
                                       rgbBlob->GetDataPtr(),rgbType,rgbBlob); // rgbTensor.IsContiguous() = true
    o3d_tensor::Image imageO3d(rgbTensor);
    auto depthBlob = std::make_shared<o3d_core::Blob>(
            device_gpu, deviceDepth.cudaPtr(), nullptr);
    auto depthTensor = o3d_core::Tensor({row,col,1}, {col,1,1},
                                        depthBlob->GetDataPtr(),depthType,depthBlob);
    o3d_tensor::Image depthO3d(depthTensor);
    /**
     * The below test is whether I can use .To(cpu)...
     */
    o3d_tensor::Image render;




    // TSDF volume
   auto intrinsic = zedParam.getO3dIntrinsicTensor();
   auto volumePtr = (new o3d_tensor::TSDFVoxelGrid({{"tsdf", open3d::core::Dtype::Float32},
                                                {"weight", open3d::core::Dtype::UInt16},
                                                {"color", open3d::core::Dtype::UInt16}},
                                               8.f / 512.f,  0.04f, 16,
                                               1000, device_gpu));


   auto extrinsicO3dTensor = o3d_core::Tensor::Eye(4,o3d_core::Float64,device_gpu); // todo from ZED
   shared_ptr<o3d_legacy::Image> renderImagePtr = make_shared<o3d_legacy::Image>();
   shared_ptr<o3d_legacy::TriangleMesh> meshPtr = make_shared<o3d_legacy::TriangleMesh>();
   o3d_tensor::TriangleMesh mesh;
   o3d_vis::Visualizer vis;
   vis.CreateVisualizerWindow("Render Image");

    while (true){
        zedState.grab(zedParam);
        cv::cuda::cvtColor(deviceImage, deviceImage3ch, cv::COLOR_BGRA2RGB);

        // mesh rendering
        {

            ElapseMonitor monitor("TSDF integration + register mesh"); // in Release, 2ms
            volumePtr->Integrate(depthO3d_cpu,imageO3d_cpu,intrinsic,extrinsicO3dTensor,1,5);


             mesh = volumePtr->ExtractSurfaceMesh(-1,3.0,
                                                 o3d_tensor::TSDFVoxelGrid::SurfaceMaskCode::VertexMap |
                                                 o3d_tensor::TSDFVoxelGrid::SurfaceMaskCode::ColorMap);
        }

        *meshPtr = mesh.ToLegacy();
        if (meshPtr->HasTriangles()){
            if (! isVisInit){
                vis.AddGeometry(meshPtr);
                isVisInit = true;
            }else{
                vis.UpdateGeometry(meshPtr);
                vis.PollEvents();
                vis.UpdateRender();
            }
        }

    }

    return 0;
}