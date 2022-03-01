//
// Created by junbs on 2/8/2022.
//
#include <SceneInterpreter.h>
#include <gtest/gtest.h>

string configFile="";
using namespace iswy;

class CameraThread : public ::testing::Test {
protected:
    SceneInterpreter* scenePtr;
    void SetUp() override {
        misc::ElapseMonitor monitor("SVO opening");
        ASSERT_NO_THROW(scenePtr = new SceneInterpreter(configFile));
    }

    void TearDown( ) override {
    }
};

TEST_F(CameraThread, SVO_OPEN) {
    try {
        cout << "Camera open success !" << endl;
    } catch (std::exception const & err) {
        FAIL() << err.what();
    }
}

// Run zed camera for rgb + depth and inspect open3d image to check binding
TEST_F(CameraThread, RUN_CAM) {
    scenePtr->disableObjectDetection();
    try {
        auto  imageO3dPtr = make_shared<open3d::geometry::Image>();
        open3d::visualization::Visualizer vis;
        vis.CreateVisualizerWindow("Open3D image",720,404);
        bool isVisInit = false;

        cout << "Running camera thread for 20 s...." << endl;
        auto timer = misc::Timer();
        float avgElapse = 0;
        int cnt = 0;
        while (timer.stop() < 20 * 1E+3){
            {
                misc::Timer tic;
                if (!scenePtr->grab()) {
                    FAIL();
                }
                avgElapse += tic.stop();
                cnt ++;

                *imageO3dPtr = scenePtr->getImageO3d();
                if (! isVisInit) {
                    vis.AddGeometry(imageO3dPtr);
                    isVisInit = true;
                } else {
                    vis.UpdateGeometry();
                    vis.PollEvents();
                    vis.UpdateRender();
                }
            }
            sl::sleep_ms(20);
        }
        cout << "average cam grab = " << (avgElapse/cnt) << " ms" << endl;
        SUCCEED();
    } catch (std::exception const & err) {
        FAIL() << err.what();
    }
}

// Run yolo object detections
TEST_F(CameraThread, OBJECTS_DETECT) {
    try {
        cout << "Camera open success !" << endl;
        cout << "Running camera thread for 20 s...." << endl;
        auto timer = misc::Timer();
        float avgElapse = 0;
        int cnt = 0;
        while (timer.stop() < 20 * 1E+3){
            misc::Timer tic;
            if (!scenePtr->grab()) {
                FAIL();
            }
            avgElapse += tic.stop();
            cnt ++;
            sl::sleep_ms(1);
        }
        cout << "average cam grab + objects detection = " << (avgElapse/cnt) << " ms" << endl;
        SUCCEED();
    } catch (std::exception const & err) {
        FAIL() << err.what();
    }
}

// Rigging skeletal bones
TEST_F(CameraThread, RIGGING) {
    namespace o3d_legacy = open3d::geometry;
    namespace o3d_core = open3d::core;
    namespace o3d_vis = open3d::visualization;

    scenePtr->disableObjectDetection(); // don't need object detection. We only test actor.
    try {
        cout << "Camera open success !" << endl;
        cout << "Running camera thread for 20 s...." << endl;

        open3d::visualization::Visualizer vis;
        vis.CreateVisualizerWindow("Rigging",720,404);
        std::shared_ptr<o3d_legacy::LineSet> skeletonO3dPtr =
                std::make_shared<o3d_legacy::LineSet>() ; // object skeleton
        bool isVisInit = false;
        vis.GetRenderOption().ChangeLineWidth(1.0);

        auto timer = misc::Timer();
        int cnt = 0;
        while (timer.stop() < 20 * 1E+3){

                if (!scenePtr->grab())
                    FAIL();

                // if success, visualize
                *skeletonO3dPtr = scenePtr->getSkeletonO3d();
                if (! skeletonO3dPtr->IsEmpty())
                    if (! isVisInit) {
                        vis.AddGeometry(skeletonO3dPtr);
                        isVisInit = true;
                    } else {
                        vis.UpdateGeometry();
                        vis.PollEvents();
                        vis.UpdateRender();
                    }
                else
                    cout<<"still no actor identified" << endl;

            sl::sleep_ms(1);
        }


    }catch (std::exception const & err) {
        FAIL() << err.what();
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
