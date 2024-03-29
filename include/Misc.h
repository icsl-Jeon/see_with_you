//
// Created by jbs on 21. 9. 6..
//

#ifndef ZED_OPEN3D_MISC_H
#define ZED_OPEN3D_MISC_H

#define UNCLASSIFIED -1
#define NOISE -2
#define CLUSTER_SUCCESS 0
#define FAILURE -3

#include <chrono>
#include <vector>
#include <cmath>
#include <string>
#include <map>
#include <sstream>
#include <signal.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"

using namespace std::chrono;
using namespace std;



namespace error{

    struct ZedException : public std::exception
    {
        std::string s;
        ZedException(std::string ss) : s(ss) {}
        ~ZedException() throw () {} // Updated
        const char* what() const throw() { return s.c_str(); }
    };

}

namespace misc {


    class Timer {
        steady_clock::time_point t0;
        double measuredTime;
    public:
        Timer() { t0 = steady_clock::now(); }

        double stop(bool isMillisecond = true) {
            if (isMillisecond)
                measuredTime = duration_cast<milliseconds>(steady_clock::now() - t0).count();
            else
                measuredTime = duration_cast<microseconds>(steady_clock::now() - t0).count();
            return measuredTime;
        }
    };

    template <typename T>
    std::string to_string_with_precision(const T a_value, const int n = 6){
        std::ostringstream out;
        out.precision(n);
        out << std::fixed << a_value;
        return out.str();
    }

    int const clusterColors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };

    typedef struct Point_
    {
        float depth;
        uint u ; // pixel index
        uint v ; // pixel index
        int clusterID;  // clustered ID
    }DepthPoint;

    class DBSCAN {
    public:
        DBSCAN(unsigned int minPts, float eps, vector<DepthPoint> points){
            m_minPoints = minPts;
            m_epsilon = eps;
            m_points = points;
            m_pointSize = points.size();
        }
        ~DBSCAN(){}

        int run();
        vector<int> calculateCluster(DepthPoint point);
        int expandCluster(DepthPoint point, int clusterID);
        inline double calculateDistance(const DepthPoint& pointCore, const DepthPoint& pointTarget);

        int getTotalPointSize() {return m_pointSize;}
        int getMinimumClusterSize() {return m_minPoints;}
        int getEpsilonSize() {return m_epsilon;}
        int getNCluster() {return n_totalCluster; };
        int getNNoise();

    public:
        vector<DepthPoint> m_points;

    private:
        unsigned int m_pointSize;
        unsigned int m_minPoints;
        float m_epsilon;
        unsigned int n_totalCluster;
    };
    void drawPred(string className, float conf,  float x, float y, float z,
                  int left, int top, int right, int bottom, cv::Mat &frame) ;

    struct Pose{
        Eigen::Transform<float,3,Eigen::Affine> poseMat;


    };

    struct PoseSet{
    public:
        std::vector<Pose> poseSet;
        std::string name = "PoseSet";
        PoseSet() = default;
        PoseSet(int N) {poseSet = std::vector<Pose>(N);};

        void push_back(const Pose& pose){poseSet.emplace_back(pose);}
        void set(unsigned int idx,const Pose& pose ) {poseSet[idx] = pose; }
        Pose get(unsigned int idx) const {return poseSet[idx];}
        unsigned int size() const {return poseSet.size();}

    };

    struct ElapseMonitor {
        static map<string,float> monitorResult;
        float elapseMeasured =-1 ;
        string tag;
        misc::Timer* timerPtr;

        ElapseMonitor(string tag): tag(tag) {
            timerPtr = new misc::Timer();
        }
        ~ElapseMonitor();
    };

}

#endif //ZED_OPEN3D_MISC_H
