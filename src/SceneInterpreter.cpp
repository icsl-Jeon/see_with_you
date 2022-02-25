//
// Created by jbs on 21. 10. 31..
//

#include <SceneInterpreter.h>



namespace iswy{

    void Camera::bindHost() {
        try {
            uint row = zedParam.getCvSize().height;
            uint col = zedParam.getCvSize().width;

            hostData.depthCv = zed_utils::slMat2cvMat(zedState.depth); // bound buffer
            hostData.imageCv = zed_utils::slMat2cvMat(zedState.image);
            hostData.bgDepthCv = cv::Mat(hostData.depthCv.size(), CV_32FC1); // pixels of obj + human = NAN
            hostData.imageCv3ch = cv::Mat(zedParam.getCvSize(), CV_8UC3);

            // opencv gpu - open3d
            hostData.rgbBlob = std::make_shared<o3d_core::Blob>(
                    hostData.device_cpu, hostData.imageCv3ch.data, nullptr);
            hostData.rgbTensor = o3d_core::Tensor({row, col, 3}, {3 * col, 3, 1},
                                                  hostData.rgbBlob->GetDataPtr(), hostData.rgbType,
                                                  hostData.rgbBlob); // rgbTensor.IsContiguous() = true
            hostData.imageO3d = o3d_tensor::Image(hostData.rgbTensor);

            hostData.depthBlob = std::make_shared<o3d_core::Blob>(
                    hostData.device_cpu, hostData.depthCv.data, nullptr);
            hostData.depthTensor = o3d_core::Tensor({row, col, 1}, {col, 1, 1},
                                                    hostData.depthBlob->GetDataPtr(), hostData.depthType,
                                                    hostData.depthBlob);
            hostData.depthO3d = o3d_tensor::Image(hostData.depthTensor);

            // print status
            int rowCheck1 = hostData.imageO3d.GetRows();
            int colCheck1 = hostData.imageO3d.GetCols();
            int rowCheck2 = hostData.depthO3d.GetRows();
            int colCheck2 = hostData.depthO3d.GetCols();
            int imageChCheck = hostData.imageO3d.GetChannels();

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

        // create cpu memory for opencv and open3d
        bindHost();
    }

    bool Camera::grab() {
        if (!isBound) {
            cerr << "Run Camera::grab() after calling bindDevice()" << endl;
            return false;
        }
        zedState.grab(zedParam);
        cv::cvtColor(hostData.imageCv, hostData.imageCv3ch, cv::COLOR_BGRA2RGB);

        isGrab = true;
        return true;
    }

    SceneInterpreter::SceneInterpreter(string configFile) : camera(configFile), detector(configFile){
        printf("Scene interpreter initiated.\n");
    }

    bool SceneInterpreter::grab() {
        if (! camera.grab()) {
            cerr << "camera (rgb, depth, pose, actor) grab failed" << endl;
            return false;
        }
        if (! detector.detect(camera.getImageCv())){
            return false;
        }
        return true;
    }

    cv::Mat Camera::getImageCv() const {
        return hostData.imageCv3ch.clone();
    }

    open3d::geometry::Image Camera::getImageO3d() const {
        return hostData.imageO3d.ToLegacy();

    }
    open3d::geometry::Image SceneInterpreter::getImageO3d() const {
        return camera.getImageO3d();
    }

    bool param::ObjectDetectParam::parseFrom(string configFile) {
        try{
            YAML::Node config = YAML::LoadFile(configFile);
            const auto configRoot = config["objects"];
            isActive = configRoot["active"].as<bool>();
            if (isActive) {
                if (configRoot["weight"])
                    modelWeight = configRoot["weight"].as<string>();
                if (configRoot["cfg"])
                    modelConfig = configRoot["cfg"].as<string>();
                if (configRoot["names"])
                    classNameDir = configRoot["names"].as<string>();
                if (configRoot["confidence"])
                    confidence = configRoot["confidence"].as<float>();
            }else{
                cout << "Objects detection (yolo) disabled." << endl;
            }
        } catch (YAML::Exception  &e){
            cerr << "error when opening yaml config" << endl;
        }

    }

    ObjectDetector::ObjectDetector(string configFile) {
        // parse config file
        if (! paramDetect.parseFrom(configFile))
            throw std::runtime_error("ObjectDetector initialization failed");

        // detector init
        if (paramDetect.isActive) {
            std::ifstream ifs(paramDetect.classNameDir.c_str());
            if (!ifs.is_open())
                CV_Error(cv::Error::StsError, "File " + paramDetect.classNameDir + " not found");
            std::string line;
            while (std::getline(ifs, line))
                paramDetect.classNames.push_back(line);

            net = cv::dnn::readNetFromDarknet(paramDetect.modelConfig, paramDetect.modelWeight);
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }
    }

    bool ObjectDetector::detect(const cv::Mat& imageCv3ch) {
        if (! paramDetect.isActive)
            return true;
        try {
            static cv::Mat blob, frame;
            frame = imageCv3ch.clone();
            cv::dnn::blobFromImage(frame, blob, 1.0,
                                   {608, 608}, // TODO
                                   cv::Scalar(), true, false);
            net.setInput(blob, "", 1.0 / 255);

            // inference
            std::vector<cv::Mat> outs;
            std::vector<cv::String> outNames = net.getUnconnectedOutLayersNames();
            net.forward(outs, outNames);
            return true;
        } catch (std::exception& e){
            cerr << "Exception at object detection " << e.what();
            return false;
        }
    }

    void SceneInterpreter::disableObjectDetection() {
        detector.paramDetect.isActive = false;
    }

    void SceneInterpreter::enableObjectDetection() {
        detector.paramDetect.isActive = true;
    }

    open3d::geometry::LineSet SceneInterpreter::getSkeletonO3d() const {
        sl::ObjectData human = camera.getHumanZed();
        open3d::geometry::LineSet skeleton;
        o3d_utils::fromSlObjects(human, skeleton );
        return skeleton;
    }











}