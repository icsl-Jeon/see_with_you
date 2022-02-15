//
// Created by jbs on 21. 9. 5..
//
#include <ZedUtils.h>

#include <string>
using namespace std;
using namespace zed_utils;

CameraParam::CameraParam(string parameterFilePath) {

    // parameter parsing
    try{
        YAML::Node config = YAML::LoadFile(parameterFilePath);
        if(config["svo_file"]){
            string svoFile = config["svo_file"].as<string>();

            if (! svoFile.empty()) {
                cout << "SVO file given: " << svoFile << endl;
                initParameters->input.setFromSVOFile(svoFile.c_str());
                isSvo = true;
            }
            else
                initParameters->camera_resolution = sl::RESOLUTION::HD720; // TODO: param

        }else
            initParameters->camera_resolution = sl::RESOLUTION::HD720;
    } catch (YAML::Exception  &e){
        cerr << "error when opening yaml config" << endl;
    }

    // todo
    initParameters = new sl::InitParameters;
    detectionParameters = new sl::ObjectDetectionParameters;
    runtimeParameters = new sl::RuntimeParameters;
    objectDetectionRuntimeParameters = new sl::ObjectDetectionRuntimeParameters;
    positionalTrackingParameters = new sl::PositionalTrackingParameters;


    initParameters->coordinate_units = sl::UNIT::METER;
    initParameters->coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
    initParameters->depth_mode = sl::DEPTH_MODE::ULTRA;
    initParameters->depth_maximum_distance = 7.0;
    initParameters->depth_minimum_distance = 0.1;

    detectionParameters->detection_model = sl::DETECTION_MODEL::HUMAN_BODY_FAST;
    detectionParameters->enable_tracking = true;
    detectionParameters->enable_body_fitting = true;
    detectionParameters->enable_mask_output = true;

    runtimeParameters->confidence_threshold = 50;
    positionalTrackingParameters->enable_area_memory = true;
}

Eigen::Matrix3d CameraParam::getCameraMatrix() const {
    Eigen::Matrix3d camMat;
    camMat.setIdentity();
    camMat(0,0) = fx;
    camMat(1,1) = fy;
    camMat(0,2) = cx;
    camMat(1,2) = cy;

    return camMat;
}


bool CameraParam::open(ZedState& zed) {

    // open camera
    auto returned_state = zed.camera.open(*initParameters);
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        printf("Enabling positional tracking failed: \n");
        return false;
    }

    returned_state = zed.camera.enablePositionalTracking();
    if(returned_state != sl::ERROR_CODE::SUCCESS) {
        printf("Enabling positional tracking failed: \n");
        zed.camera.close();
        return false;
    }

    returned_state = zed.camera.enableObjectDetection(detectionParameters);
    if(returned_state != sl::ERROR_CODE::SUCCESS) {
        printf("Enabling object detection failed: \n");
        zed.camera.close();
        return false;
    }

    returned_state = zed.camera.enablePositionalTracking(*positionalTrackingParameters);
    if(returned_state != sl::ERROR_CODE::SUCCESS) {
        printf("Enabling position tracking failed: \n");
        zed.camera.close();
        return false;
    }

    sl::CameraConfiguration intrinsicParam = zed.camera.getCameraInformation().camera_configuration;
    fx = intrinsicParam.calibration_parameters.left_cam.fx;
    fy = intrinsicParam.calibration_parameters.left_cam.fy;
    cx = intrinsicParam.calibration_parameters.left_cam.cx;
    cy = intrinsicParam.calibration_parameters.left_cam.cy;
    width = zed.camera.getCameraInformation().camera_resolution.width;
    height = zed.camera.getCameraInformation().camera_resolution.height;

    // initialize image matrix
    zed.image.alloc(width,height, sl::MAT_TYPE::U8_C4,  sl::MEM::GPU);
    zed.depth.alloc(width,height, sl::MAT_TYPE::F32_C1, sl::MEM::GPU);

    return true;
}


cv::Point2f CameraParam::project(const Eigen::Vector3f& pnt) const{
    return {fx*pnt.x()/pnt.z() + cx, fx*pnt.y()/pnt.z() + cy};};

void CameraParam::unProject (cv::Point uv, float depth, float& xOut, float& yOut, float & zOut) const{
    zOut = depth;
    xOut = (uv.x - cx) * depth / fx;
    yOut = (uv.y - cy) * depth / fy;
};


bool ZedState::grab(const CameraParam& runParam) {

    misc::ElapseMonitor monitor("Grab {rgb,depth,object}");

    bool isOk;
    auto rtParam = runParam.getRtParam();
    if (camera.grab(rtParam) == sl::ERROR_CODE::SUCCESS) {
        isOk = true;
    } else if (camera.grab(rtParam) == sl::ERROR_CODE::END_OF_SVOFILE_REACHED) {
        printf("SVO reached end. Replay. \n");
        camera.setSVOPosition(1);
        isOk = true;
    } else {
        printf("Grab failed. \n");
        isOk = false;
    }

    // update states
    camera.retrieveImage(image,sl::VIEW::LEFT,sl::MEM::GPU);
    camera.retrieveMeasure(depth, sl::MEASURE::DEPTH, sl::MEM::GPU);
    camera.retrieveObjects(humans,runParam.getObjRtParam());
    camera.getPosition(pose);

    if (! humans.object_list.empty())
        actor = humans.object_list[0];

    return isOk;
}

Eigen::Matrix4f ZedState::getPoseMat() const {
    auto transform = sl::Transform();
    Eigen::Matrix<float,3,1> transl(pose.pose_data.getTranslation().v);

    Eigen::Matrix3f rot = Eigen::Map<Eigen::Matrix<float,3,3,Eigen::RowMajor> >(
            pose.pose_data.getOrientation().getRotationMatrix().r);
    Eigen::Matrix4f poseMat;
    poseMat.setIdentity();
    poseMat.block(0,0,3,3) = rot;
    poseMat.block(0,3,3,1) = transl;
    //        cout << poseMat << endl;
    return poseMat;
}

bool ZedState::markHumanPixels(cv::cuda::GpuMat &maskMat) {
    if (humans.object_list.empty())
        return false;
    auto human_bb_min = actor.bounding_box_2d[0];
    auto human_bb_max = actor.bounding_box_2d[2];
    int w = human_bb_max.x - human_bb_min.x;
    int h = human_bb_max.y - human_bb_min.y;

    cv::Mat subMaskCpu = zed_utils::slMat2cvMat(actor.mask);
    cv::cuda::GpuMat subMask; subMask.upload(subMaskCpu);
    cv::Range rowRangeHuman(human_bb_min.y,human_bb_max.y);
    cv::Range colRangeHuman(human_bb_min.x,human_bb_max.x);

    auto subPtr = maskMat(rowRangeHuman,colRangeHuman);
    subMask.copyTo(subPtr);


    return true;
}
bool zed_utils::parseArgs(int argc, char **argv,sl::InitParameters& param)
{
    bool doRecord = false;
    if (argc > 1 && string(argv[1]).find(".svo")!=string::npos) {
        // SVO input mode
        param.input.setFromSVOFile(argv[1]);
        cout<<"[Sample] Using SVO File input: "<<argv[1]<<endl;
    } else if (argc > 1 && string(argv[1]).find(".svo")==string::npos) {
        string arg = string(argv[1]);
        unsigned int a,b,c,d,port;
        if (sscanf(arg.c_str(),"%u.%u.%u.%u:%d", &a, &b, &c, &d,&port) == 5) {
            // Stream input mode - IP + port
            string ip_adress = to_string(a)+"."+to_string(b)+"."+to_string(c)+"."+to_string(d);
            param.input.setFromStream(sl::String(ip_adress.c_str()),port);
            cout<<"[Sample] Using Stream input, IP : "<<ip_adress<<", port : "<<port<<endl;
        }
        else  if (sscanf(arg.c_str(),"%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            // Stream input mode - IP only
            param.input.setFromStream(sl::String(argv[1]));
            cout<<"[Sample] Using Stream input, IP : "<<argv[1]<<endl;
        }
        else if (arg.find("HD2K")!=string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD2K;
            cout<<"[Sample] Using Camera in resolution HD2K"<<endl;
        } else if (arg.find("HD1080")!=string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD1080;
            cout<<"[Sample] Using Camera in resolution HD1080"<<endl;
        } else if (arg.find("HD720")!=string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD720;
            cout<<"[Sample] Using Camera in resolution HD720"<<endl;
        } else if (arg.find("VGA")!=string::npos) {
            param.camera_resolution = sl::RESOLUTION::VGA;
            cout<<"[Sample] Using Camera in resolution VGA"<<endl;
        }
    } else {
        // Default initialization

        doRecord = true;
    }
    return doRecord;
}

bool zed_utils::initCamera(sl::Camera &zed, sl::InitParameters initParameters) { // will be deprecated


    // Parameter setting
    initParameters.coordinate_units = sl::UNIT::METER;
    initParameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
    initParameters.depth_mode = sl::DEPTH_MODE::QUALITY;
    initParameters.depth_maximum_distance = 10.0;
    initParameters.depth_minimum_distance = 0.02;

    /**
    sl::ObjectDetectionParameters detectionParameters;
    detectionParameters.detection_model = sl::DETECTION_MODEL::HUMAN_BODY_MEDIUM;
    detectionParameters.enable_tracking = true;
    detectionParameters.enable_body_fitting = true;
    detectionParameters.enable_mask_output = true;
    **/

    // Enabling functions
    auto returned_state = zed.open(initParameters);
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        printf("Enabling positional tracking failed: \n");
        return false;
    }
    returned_state = zed.enablePositionalTracking();
    if(returned_state != sl::ERROR_CODE::SUCCESS) {
        printf("Enabling positional tracking failed: \n");
        zed.close();
        return false;
    }

    /**
    returned_state = zed.enableObjectDetection(detectionParameters);
    if(returned_state != sl::ERROR_CODE::SUCCESS) {
        printf("Enabling object detection failed: \n");
        zed.close();
        return false;
    }
    **/

    return true;
}

// Mapping between MAT_TYPE and CV_TYPE
int zed_utils::getOCVtype(sl::MAT_TYPE type) {
    int cv_type = -1;
    switch (type) {
        case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
        case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
        case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
        case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
        case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
        case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
        case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
        case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
        default: break;
    }
    return cv_type;
}

cv::Mat zed_utils::slMat2cvMat(sl::Mat& input) {
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(),
                   zed_utils::getOCVtype(input.getDataType()),
                   input.getPtr<sl::uchar1>(sl::MEM::CPU), input.getStepBytes(sl::MEM::CPU));
}


cv::cuda::GpuMat zed_utils::slMat2cvMatGPU(sl::Mat& input) {
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::cuda::GpuMat(input.getHeight(), input.getWidth(),
                            zed_utils::getOCVtype(input.getDataType()),
                            input.getPtr<sl::uchar1>(sl::MEM::GPU), input.getStepBytes(sl::MEM::GPU));
}




void zed_utils::print(string msg_prefix, ERROR_CODE err_code, string msg_suffix) {

    cout <<"[Sample]";
    if (err_code != ERROR_CODE::SUCCESS)
        cout << "[Error] ";
    else
        cout<<" ";
    cout << msg_prefix << " ";
    if (err_code != ERROR_CODE::SUCCESS) {
        cout << " | " << toString(err_code) << " : ";
        cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        cout << " " << msg_suffix;
    cout << endl;
}