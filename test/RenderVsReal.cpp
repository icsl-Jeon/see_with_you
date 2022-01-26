//
// Created by junbs on 11/21/2021.
//

#include <SceneInterpreter.h>
#include "opencv2/cudaarithm.hpp"
#include "opencv2/rgbd.hpp"
#include "iostream"

#include <RenderUtils.h> // IDK for sure, but this should be included after sl/Camera.hpp...

using namespace iswy;

namespace o3d_tensor =  open3d::t::geometry ;
namespace o3d_legacy = open3d::geometry;
namespace o3d_core = open3d::core;
namespace o3d_vis = open3d::visualization;

o3d_vis::Visualizer* vis; // This visualization should run in other thread. Avoid conviction with opengl context of our rendering algorithm
mutex mutex_;

shared_ptr<o3d_legacy::TriangleMesh> meshPtr;
shared_ptr<o3d_legacy::TriangleMesh> worldFramePtr;
std::vector<shared_ptr<o3d_legacy::TriangleMesh>> camPtrSet; // candidate cams
shared_ptr<o3d_legacy::TriangleMesh> curCamPtr; // current cam

bool isVisInit = false;
const int nCam = 10; // number of candidate views

void drawThread(){
    // We use x-fowarding z-up coordinate for camera. That is, R_wc = [0 0 1 ; -1 0 0 ; 0 -1 0]
    worldFramePtr = std::make_shared<o3d_legacy::TriangleMesh>();
    worldFramePtr = o3d_legacy::TriangleMesh::CreateCoordinateFrame(0.1);
    vis = new o3d_vis::Visualizer();
    vis->CreateVisualizerWindow("Render Image");
    while(true)
        // draw current mesh
        if (meshPtr != NULL )
            if (meshPtr->HasTriangles()  && mutex_.try_lock()   ) {
                if (!isVisInit) {
                    for (int camIdx = 0 ; camIdx < nCam ; camIdx++)
                        vis->AddGeometry(camPtrSet[camIdx]);
                    vis->AddGeometry(meshPtr);
//                    vis->AddGeometry(worldFramePtr);
                    vis->GetViewControl().SetFront({0.2,0,0.0});
                    vis->GetViewControl().SetZoom(0.5);
                    vis->GetViewControl().SetUp({0,0,1});
                    vis->GetViewControl().SetConstantZFar(100); // essential to vis. the mesh far away
                    vis->AddGeometry(curCamPtr);
                    isVisInit = true;
                } else {
                    vis->UpdateGeometry(meshPtr);
                    vis->UpdateGeometry(curCamPtr);
                    vis->PollEvents();
                    vis->UpdateRender();
                }
                mutex_.unlock();
                this_thread::sleep_for(std::chrono::milliseconds(2));
            }
}

int main(){
    string userName = "junbs"; // computer name

    string datasetDir =
            "C:\\Users\\" + userName + "\\OneDrive\\Documents\\GitHub\\edelkrone_controller\\20220121151001"; // IDK.. but double slash does not work in my desktop\\

    // OPENGL
    string shaderDir = "C:/Users/" + userName +"/OneDrive/Documents/GitHub/see_with_you/include/shader";
    render_utils::Param param;
    param.shaderRootDir = shaderDir;
    render_utils::SceneRenderServer glServer(param);
    render_utils::RenderResult renderResult;

    // Candidate cam poses reading from calibration
    camPtrSet.resize(nCam);
    misc::PoseSet camSet;
    misc::Pose poseCam; // default optical coordinate
    poseCam.poseMat.setIdentity();
    std::ifstream infile;
    infile.open(datasetDir + "\\calibration_result.txt");
    double numBuff[nCam*19];
    int insertIdx = 0;
    if(infile.is_open()){
        while(infile >> numBuff[insertIdx++]){} // read all the numeric data
        for (int camIdx = 0; camIdx < nCam ; camIdx++){
            Eigen::Map<Eigen::Matrix<double,4,4,Eigen::RowMajor> > M(&numBuff[19*camIdx+3]);
//            cout << M << endl;
            poseCam.poseMat.matrix() = M.cast<float>();
            camSet.push_back(poseCam);

            // for coord visualization in open3d
            o3d_legacy::TriangleMesh camFrame = *o3d_legacy::TriangleMesh::CreateCoordinateFrame(0.05);
            camFrame = camFrame.Transform(M);
            camPtrSet[camIdx] = std::make_shared<o3d_legacy::TriangleMesh>();
            *camPtrSet[camIdx] = camFrame;
        }
    }
    else{
        cerr <<
        "Calibration file was not opened. We require sequence of zed pose to render a mesh view from that view. Exiting.."
        << endl;
        exit(0);
    }


    // Initialize ZED

    CameraParam zedParam(" ",datasetDir + "\\record.svo");
    ZedState zedState;
    zedParam.open(zedState);

    if (zedState.camera.isOpened()){
        std::cout << "Camera opened !" << std::endl;
    }else{
        std::cerr << "Camera not opened" << std::endl;
        return 0;
    }

    // Opencv image bound for the zed grab
    cv::cuda::GpuMat deviceImage,deviceDepth,deviceImage3ch;
    deviceImage = zed_utils::slMat2cvMatGPU(zedState.image);
    deviceDepth = zed_utils::slMat2cvMatGPU(zedState.depth);
    deviceImage3ch = cv::cuda::createContinuous(zedParam.getCvSize(),CV_8UC3);
    uint row = zedParam.getCvSize().height;
    uint col = zedParam.getCvSize().width;
    cv::namedWindow("render window",cv::WINDOW_KEEPRATIO);
    cv::resizeWindow("render window", 600,400);


    cv::Mat hostDepth;

    // Open3D image binding to opencv image
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


    // TSDF volume for mesh generation (tuning: http://www.open3d.org/docs/0.12.0/tutorial/pipelines/rgbd_integration.html)
   auto intrinsic = zedParam.getO3dIntrinsicTensor();
   auto volumePtr = (new o3d_tensor::TSDFVoxelGrid({{"tsdf", open3d::core::Dtype::Float32},
                                                {"weight", open3d::core::Dtype::UInt16},
                                                {"color", open3d::core::Dtype::UInt16}},
                                               0.05,  0.04f, 16,
                                               100, device_gpu));


   auto extrinsicO3dTensor = o3d_core::Tensor::Eye(4,o3d_core::Float64,device_gpu); // todo from ZED
   shared_ptr<o3d_legacy::Image> renderImagePtr = make_shared<o3d_legacy::Image>();
   meshPtr = make_shared<o3d_legacy::TriangleMesh>();
    o3d_tensor::TriangleMesh mesh;
    curCamPtr = make_shared<o3d_legacy::TriangleMesh>();

    // Open3d visualizer
   std::thread drawRoutine(drawThread);


    // T_cw / T_wc
    Eigen::Matrix4f T_co = Eigen::Matrix4f::Zero(); // cam to optical
    T_co(0,2) = 1; T_co(1,0) = -1; T_co(2,1) = -1; T_co(3,3) = 1;


    while (true){
        zedState.grab(zedParam);
        // get pose
        Eigen::Matrix4f T_wc = zedState.getPoseMat()*T_co;
        *curCamPtr = *o3d_legacy::TriangleMesh::CreateCoordinateFrame(0.1);
        *curCamPtr = curCamPtr->Transform(T_wc.cast<double>());
        o3d_core::Tensor T_wc_t = o3d_core::eigen_converter::EigenMatrixToTensor(T_wc); T_wc_t.To(device_gpu);
        extrinsicO3dTensor = T_wc_t.Inverse(); // seems that open3d fuckers don't know what extrinsic is. Why fucking inverse is needed here?

        cv::cuda::cvtColor(deviceImage, deviceImage3ch, cv::COLOR_BGRA2RGB);

        {
            // mesh rendering
            {
                ElapseMonitor monitor("TSDF integration + register mesh ");
                volumePtr->Integrate(depthO3d,imageO3d,intrinsic,extrinsicO3dTensor,1,10);


                mesh = volumePtr->ExtractSurfaceMesh(-1,1.0,
                                                     o3d_tensor::TSDFVoxelGrid::SurfaceMaskCode::VertexMap |
                                                     o3d_tensor::TSDFVoxelGrid::SurfaceMaskCode::ColorMap);
                mesh = mesh.To(device_cpu);
            }
            if (! mesh.IsEmpty()) {
                renderResult = glServer.renderService(camSet, mesh);
                cv::Mat renderImage = renderResult.renderImage;
                cv::imshow("render window", renderImage);
                cv::waitKey(1);
            }

        }
        {
            lock_guard<mutex> lock(mutex_);
            *meshPtr = mesh.ToLegacy();
        }
    }

    drawRoutine.join();

    return 0;
}