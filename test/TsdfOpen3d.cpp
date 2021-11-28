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


    deviceImage3ch.download(hostImage);
    o3d_core::Tensor tensorRgbDummy(
            (hostImage.data), {row,col,3},rgbType,device_cpu);
    tensorRgbDummy.To(device_gpu);

    auto rgbBlob = std::make_shared<o3d_core::Blob>(
            device_gpu,deviceImage3ch.cudaPtr(), nullptr);
    auto  rgbTensor = o3d_core::Tensor({row,col,3},{3*col,3,1},
                                       rgbBlob->GetDataPtr(),rgbType,rgbBlob); // rgbTensor.IsContiguous() = true
    o3d_tensor::Image imageO3d = o3d_tensor::Image(rgbTensor);

    auto depthBlob = std::make_shared<o3d_core::Blob>(
            device_gpu, deviceDepth.cudaPtr(), nullptr);
    auto depthTensor = o3d_core::Tensor({row,col,1}, {col,1,1},
                                        depthBlob->GetDataPtr(),depthType,depthBlob);

    o3d_tensor::Image depthO3d(depthTensor);
    o3d_tensor::Image render;

    cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, depthTensor.GetDataPtr());

    // TSDF volume
   auto intrinsic = zedParam.getO3dIntrinsicTensor();
   auto volumePtr = (new o3d_tensor::VoxelBlockGrid( {"tsdf","weight","color"} ,
                                     {o3d_core::Dtype::Float32,o3d_core::Dtype::UInt16,o3d_core::Dtype::UInt16},
                                     {{1},{1},{3}},
                                      8.f / 512.f,  16,
                                      1000, device_cpu));


   auto extrinsicO3dTensor = o3d_core::Tensor::Eye(4,o3d_core::Float64,device_gpu); // todo from ZED
   shared_ptr<o3d_legacy::Image> renderImagePtr = make_shared<o3d_legacy::Image>();
   shared_ptr<o3d_legacy::TriangleMesh> meshPtr = make_shared<o3d_legacy::TriangleMesh>();
   o3d_tensor::TriangleMesh mesh;
   o3d_vis::Visualizer vis;
   vis.CreateVisualizerWindow("Render Image");


    auto  imageO3dPtr = std::make_shared<open3d::geometry::Image>();
    auto  depthO3dPtr = std::make_shared<open3d::geometry::Image>();
    imageO3dPtr->Prepare(col,row,
                         3,1); // 8U_3C
    depthO3dPtr->Prepare(col,row,
                         1, 4); // 32F_1C

    /**
     * In case of rendering of TSDF mesh, I have explored three:
     * 1) RayCast in TSDFVolume
     *
            auto results = volumePtr->RayCast(intrinsic,extrinsicO3dTensor,col*2,row*2,1,0.01,5.0,0,
                                              o3d_tensor::TSDFVoxelGrid::SurfaceMaskCode::ColorMap);
            render = o3d_tensor::Image(results[o3d_tensor::TSDFVoxelGrid::ColorMap]).To(device_cpu);

            For this, we cannot adjust the render image resolution. Useless

     * -> Image was cracked. I opened an issue (https://github.com/isl-org/Open3D/issues/4340)
     *
     * 2) RayCastingScene class (ref http://www.open3d.org/docs/latest/tutorial/geometry/ray_casting.html) : seems not to support RGB
     *
     * 3) Offscreen rendering : not possible to run headless and non-headless at the same time (https://github.com/isl-org/Open3D/issues/2998)
     **/
    while (true){

        zedState.grab(zedParam);
        cv::cuda::cvtColor(deviceImage, deviceImage3ch, cv::COLOR_BGRA2RGB);

        deviceDepth.download(hostDepth);
        deviceImage3ch.download(hostImage);

        // mesh rendering
        intrinsic = intrinsic.To(device_cpu);
        extrinsicO3dTensor = extrinsicO3dTensor.To(device_cpu);

        o3d_utils::fromCvMat(hostImage,*imageO3dPtr);
        o3d_utils::fromCvMat(hostDepth,*depthO3dPtr);

        o3d_tensor::Image imageO3d_cpu, depthO3d_cpu;
        imageO3d_cpu = imageO3d_cpu.FromLegacy(*imageO3dPtr);
        depthO3d_cpu = depthO3d_cpu.FromLegacy(*depthO3dPtr);

        auto blockCoord = volumePtr->GetUniqueBlockCoordinates(depthO3d_cpu,intrinsic,extrinsicO3dTensor,1,5);
        blockCoord = blockCoord.To(device_cpu);
//        ElapseMonitor monitor("TSDF integration + register mesh"); // in Release, 2ms
        volumePtr->Integrate(blockCoord,depthO3d_cpu,imageO3d_cpu,intrinsic,extrinsicO3dTensor,1,5);

        // mesh = volumePtr->ExtractTriangleMesh(-1,3.0);
        // *meshPtr = mesh.ToLegacy();
        /**
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
        **/

    }

    return 0;
}