//
// Created by JBS on 12/6/2021.
//

#ifndef SEE_WITH_YOU_RENDERUTILS_H
#define SEE_WITH_YOU_RENDERUTILS_H


#include <shader/shader.h>
#include <opencv2/opencv.hpp>
#include <Misc.h>
#include <cstdlib>
#include <open3d/Open3D.h>
#include <glm/glm.hpp>


using namespace std;
using namespace misc;

namespace o3d_tensor =  open3d::t::geometry ;
namespace o3d_legacy = open3d::geometry;
namespace o3d_core = open3d::core;
namespace o3d_vis = open3d::visualization;

const int nMaxTriangles = 300000;


namespace render_utils {
    enum RenderTileFormat {
        DISPLAY_RATIO, // arrange as close as possible to the DISPLAY RATIO
    };

    struct RenderArgument {
        int nTriangle; // number of triangles
        unsigned int tileWidth; // width per image (e.g. 640 x 480)
        unsigned int tileHeight; // height per image
        unsigned int renderFov;
        unsigned int vao; // Vertex Array Object
        unsigned int quadVao;
        unsigned int quadVbo;
        unsigned int frameBuffer;
        unsigned int textureColorBuffer;
        unsigned int tileRow;
        unsigned int tileCol;
    };

    struct Param {
        bool verbose;
        // tile size.
        unsigned int imgWidth = 300;
        unsigned int imgHeight = 300;
        // total composite image
        unsigned int displayWidth = 1920;
        unsigned int displayHeight = 1080;
        float fovDeg = 90.0f; // degree
        string shaderRootDir;
    };

    //! Arrange tiles (render scene from multiple poses) into suitable row and column
    void arrangeTile(RenderTileFormat format, unsigned int displayW, unsigned int displayH,
                     unsigned int tileW, unsigned int tileH,
                     unsigned int numCam, unsigned int &R, unsigned int &C);

    glm::mat4 toGlm(const misc::Pose& pose);

    //! Initialize shader projection (P) / model matrix (M)
    void initShaderTransform(Shader *shader, float FOV, int width, int height);
    //! Initialize shader view (V) with request camPose
    void setCameraSingle(Shader *shader, const misc::Pose &camPose);
    //! Read current image from graphic buffer
    cv::Mat readToCv(int width, int height);
    //! Bind vertex and shade, outputing cv::Mat
    cv::Mat render(Shader *shader, const misc::Pose &camPose, RenderArgument arg, vector<cv::Rect> &tileRCMap);
    //! Shade tile (R x C) using scene shading
    cv::Mat renderTile(Shader *shader, Shader *shaderTile,
                       const PoseSet &poseSet, RenderArgument arg, vector<cv::Rect> &tileRCMap,
                       float * vertexTile);

    struct RenderResult{
        unsigned int tileRow;
        unsigned int tileCol;
        vector<cv::Rect> tileMap; //! each Rect is associated tile region
        cv::Mat renderImage; //! raw rendering output from tile shader
        PoseSet viewPoseSet;
        bool isRenderSuccess;
        double elapse;
        vector<cv::Mat> divideImages() const;

    };

    class SceneRenderServer  {

        struct OpenglObjects {
            // scene shader
            unsigned int VBO;
            unsigned int VAO;
            unsigned int EBO;

            // screen shader (tile compiler)
            unsigned int quadVBO;
            unsigned int quadVAO;

            // frame buffer
            unsigned int framebuffer;
            unsigned int textureColorBuffer;
            unsigned int rbo;
        };

    private:
        Param param;
        bool isShaderInit = false;
        bool isRenderTried = false;
        string myName = "MeshShader: ";
        OpenglObjects openglObjs; //! buffer objects
        Shader *sceneShader; //! render scene with meshes from a camera pose
        Shader *tileShader; //! augment scenes from multiple camera pose in GPU side

        bool createOpenglContext(); //! This should be called in the same thread where render is requested
        bool initShader(string shaderRootDir); //! vertex and fragment shader init

        //! Raw buffer used for opengl rendering. This should be built from mesh queue at meshQueueToVertexBuffer()
        float* vertexPtr; // x,y,z,r,g,b
        unsigned int* indexPtr; // v1,v2,v3
        float* vertexTile;
        int nVertex = 0; //! number of vertex (x,y,z,r,g,b). total data array to contain = 6*nVertex Opengl VBO uses this
        int nTriIndex = 0; //! number of vertex (v1,v2,v3). total data array to contain = 3*nTriIndex Opengl EBO uses this

        //! Bind current vertexPtr and indexPtr
        void uploadVertexToRender();

        int getNumVertex() const { return nVertex; }
        int getNumIndex() const { return nTriIndex; };
        int nMesh() const { return nTriIndex; }

        //! Save meshes from voxblox to vertex buffer for opengl rendering
        void insertVertex(float x, float y, float z, float r, float g, float b);
        //! Enumerate vertex for triangle interpretation
        void insertIndex(unsigned int vertex1, unsigned int vertex2, unsigned int vertex3);
        void printStr (string msg) {cout << myName << msg << endl; }
        //! Core function. Read meshes and update to vertex and index array, binding to opengl
        bool uploadMesh(const o3d_tensor::TriangleMesh& mesh);

    public:
        SceneRenderServer()  = default;
        SceneRenderServer(const Param &param) : param(param){  initShader(param.shaderRootDir);  };
        //! Core function. Perform rendering given set of cam pose  and meshes
        RenderResult renderService(const PoseSet& camPoseSet, const o3d_tensor::TriangleMesh& mesh);


    };

};
#endif //SEE_WITH_YOU_RENDERUTILS_H
