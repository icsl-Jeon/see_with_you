//
// Created by junbs on 2/8/2022.
//
#include <ZedUtils.h>
#include <gtest/gtest.h>

string configFile="";
TEST(ZED_WRAPPER, SVO_OPEN) {
    try {
        zed_utils::CameraParam camParam(configFile);
        SUCCEED();
    } catch (std::exception const & err) {
        FAIL() << err.what();
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc,argv);
    if (argc > 1){
        configFile = string(argv[1]);
    }else{
        cerr << "no config file given. Provide config.yaml" << endl;
        return 0;
    }

    return RUN_ALL_TESTS();
}
