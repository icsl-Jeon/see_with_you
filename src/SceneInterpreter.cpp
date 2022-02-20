//
// Created by jbs on 21. 10. 31..
//

#include <SceneInterpreter.h>


map<string,float> misc::ElapseMonitor::monitorResult = map<string,float>();

namespace iswy{

    void Camera::bindDevice() {
        try {
            uint row = zedParam.getCvSize().height;
            uint col = zedParam.getCvSize().width;

            // zed - opencv gpu
            deviceData.depthCv = zed_utils::slMat2cvMatGPU(zedState.depth); // bound buffer
            deviceData.imageCv = zed_utils::slMat2cvMatGPU(zedState.image);
            deviceData.bgDepthCv = cv::cuda::createContinuous(deviceData.depthCv.size(),
                                                              CV_32FC1); // pixels of obj + human = NAN
            deviceData.imageCv3ch = cv::cuda::createContinuous(zedParam.getCvSize(), CV_8UC3);

            // opencv gpu - open3d
            deviceData.rgbBlob = std::make_shared<o3d_core::Blob>(
                    deviceData.device_gpu, deviceData.imageCv3ch.cudaPtr(), nullptr);
            deviceData.rgbTensor = o3d_core::Tensor({row, col, 3}, {3 * col, 3, 1},
                                                    deviceData.rgbBlob->GetDataPtr(), deviceData.rgbType,
                                                    deviceData.rgbBlob); // rgbTensor.IsContiguous() = true
            deviceData.imageO3d = o3d_tensor::Image(deviceData.rgbTensor);

            deviceData.depthBlob = std::make_shared<o3d_core::Blob>(
                    deviceData.device_gpu, deviceData.depthCv.cudaPtr(), nullptr);
            deviceData.depthTensor = o3d_core::Tensor({row, col, 1}, {col, 1, 1},
                                                      deviceData.depthBlob->GetDataPtr(), deviceData.depthType,
                                                      deviceData.depthBlob);
            deviceData.depthO3d = o3d_tensor::Image(deviceData.depthTensor);

            // print status
            int rowCheck1 = deviceData.imageO3d.GetRows();
            int colCheck1 = deviceData.imageO3d.GetCols();
            int rowCheck2 = deviceData.depthO3d.GetRows();
            int colCheck2 = deviceData.depthO3d.GetCols();
            int imageChCheck = deviceData.imageO3d.GetChannels();

            if ((rowCheck1 == rowCheck2) && (colCheck1 == colCheck2) && (rowCheck1 > 0) && (colCheck1 > 0)) {
                printf("Binding success. camera image (r,c) = (%d, %d) with ch = %d \n",rowCheck1,colCheck1, imageChCheck);
                isBound = true;
            }else{
                printf("Binding failed. \n");
                throw std::runtime_error("Binding in gpu memory failed.");
            }

        } catch (std::exception& e){
            cerr << "Exception at binding devices: " << e.what();
        }
    }

    Camera::Camera(string configFile): zedParam(configFile){
        // open zed camera with initial parameter
        if (! zedParam.init(zedState)){
            throw std::runtime_error("Camera initialization failed.");
            return;
        }

        // create gpu memory for opencv and open3d
        bindDevice();
    }

    bool Camera::grab() {
        if (!isBound) {
            cerr << "Run Camera::grab() after calling bindDevice()" << endl;
            return false;
        }
        zedState.grab(zedParam);
        cv::cuda::cvtColor(deviceData.imageCv, deviceData.imageCv3ch, cv::COLOR_BGRA2RGB);
        isGrab = true;
        return true;
    }

    SceneInterpreter::SceneInterpreter(string configFile) : camera(configFile){
        printf("Scene interpreter initiated.\n");
    }

    bool SceneInterpreter::grab() {
        if (! camera.grab()) {
            cerr << "camera (rgb, depth, pose, actor) grab failed" << endl;
            return false;
        }

        return true;
    }

    open3d::geometry::Image SceneInterpreter::getImageO3d() const {
        return camera.deviceData.imageO3d.ToLegacy();
    }











}