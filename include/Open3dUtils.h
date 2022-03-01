//
// Created by jbs on 21. 9. 6..
//

#ifndef ZED_OPEN3D_OPEN3DUTILS_H
#define ZED_OPEN3D_OPEN3DUTILS_H

#include <open3d/Open3D.h>
#include <open3d/t/geometry/RaycastingScene.h>
#include <open3d/core/MemoryManager.h>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp> // This should not be double included ..?

using namespace std;
namespace o3d_utils {
    using namespace sl;

    void fromCvMat(const cv::Mat& cvImage, open3d::geometry::Image& o3dImage );
    void fromSlPoints(const sl::Mat& slPoints, open3d::geometry::PointCloud& o3dPoints );
    void fromSlObjects(const sl::ObjectData& object,
                       open3d::geometry::LineSet& skeletonLineSet
                       );
    void initViewport(open3d::visualization::Visualizer& vis); // todo parameterized
    void registerGeometrySet(open3d::visualization::Visualizer& vis,
                             const vector<shared_ptr<open3d::geometry::Geometry3D>>& geometryPtrSet) ;



}

#endif //ZED_OPEN3D_OPEN3DUTILS_H
