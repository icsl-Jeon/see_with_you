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
