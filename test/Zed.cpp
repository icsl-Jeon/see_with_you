//
// Created by junbs on 2/8/2022.
//
#include <ZedUtils.h>
#include <gtest/gtest.h>

string configFile="";


class ZedOpen : public ::testing::Test {
protected:
    zed_utils::CameraParam* initParamPtr;
    zed_utils::ZedState* zedStatePtr;
    void SetUp() override {
        initParamPtr = new zed_utils::CameraParam(configFile);
        zedStatePtr = new zed_utils::ZedState;
        initParamPtr->init(*zedStatePtr);
    }

    void TearDown( ) override {
        delete initParamPtr;
        delete zedStatePtr;
    }
};

TEST_F(ZedOpen, SVO_OPEN) {
    try {
        misc::ElapseMonitor monitor("SVO opening");
        ASSERT_EQ(zedStatePtr->isCameraOpen(), 1);
    } catch (std::exception const & err) {
        FAIL() << err.what();
    }
}

TEST_F(ZedOpen, BODY_TRACKING_LEGACY){
    using namespace sl;
    Camera zed;
    InitParameters init_parameters;
    init_parameters.camera_resolution = RESOLUTION::HD1080;
    // On Jetson the object detection combined with an heavy depth mode could reduce the frame rate too much
    init_parameters.depth_mode = DEPTH_MODE::ULTRA;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    init_parameters.input.setFromSVOFile("C:/Users/JBS/OneDrive/Documents/ZED/HD1080_SN28007858_14-25-48.svo");

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        zed.close();
    }

    PositionalTrackingParameters positional_tracking_parameters;

    returned_state = zed.enablePositionalTracking(positional_tracking_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        zed.close();
    }

    // Enable the Objects detection module
    ObjectDetectionParameters obj_det_params;
    obj_det_params.enable_tracking = true; // track people across images flow
    obj_det_params.enable_body_fitting = true; // smooth skeletons moves
    obj_det_params.body_format = sl::BODY_FORMAT::POSE_34;
    obj_det_params.detection_model = DETECTION_MODEL::HUMAN_BODY_FAST ;
    returned_state = zed.enableObjectDetection(obj_det_params);
    if (returned_state != ERROR_CODE::SUCCESS) {
        zed.close();
    }


    // Create ZED Objects filled in the main loop
    Objects bodies;
    char key = ' ';

    while (key != 'q') {
        // Grab images
        if (zed.grab() == ERROR_CODE::SUCCESS) {
            // Once the camera has started, get the floor plane to stick the bounding box to the floor plane.

            // Retrieve Detected Human Bodies
            auto objectTracker_parameters_rt = initParamPtr->getObjRtParam();
            zed.retrieveObjects(bodies, objectTracker_parameters_rt);
            cout << bodies.object_list[0].keypoint.size() << endl;
            string window_name = "ZED| 2D View";
            key = cv::waitKey(10);
        }
        else
            cout << "grab failed" << endl;
    }
}


TEST_F(ZedOpen, BODY_TRACKING){

    char key = ' ';
    while (key!='q'){
        if (zedStatePtr->grab(*initParamPtr)){
            cout << zedStatePtr->actor.keypoint.size() << endl;
            key = cv::waitKey(10);
        }else
            cout << "grab failed" << endl;

    }
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc,argv);
    if (argc > 1){
        configFile = string(argv[1]);
    }else{
        cerr << "no config file given. For gtesting, provide config.yaml" << endl;
        return 0;
    }

    return RUN_ALL_TESTS();
}
