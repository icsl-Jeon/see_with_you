//
// Created by JBS on 12/6/2021.
//

#include <RenderUtils.h>

using namespace std;
using namespace misc;

namespace render_utils{
    /*
        * compute suitable tile arrangement (rows and cols)
        * @param format @ref RenderTileFormat
        * @param displayW the display width of total render outoput
        * @param displayH the display width of total render outoput
        * @param tileW single tile image width
        * @param tileH single tile height
        * @param numCam total number of tiles
        * @param R output rows
        * @param C output cols
        * @todo currently only display ratio was implemented
    */
    void arrangeTile(RenderTileFormat format, unsigned int tileW, unsigned int tileH,
                          unsigned int displayW, unsigned int displayH,
                          unsigned int numCam, unsigned int &R, unsigned int &C) {

        unsigned int Rmax = floor( displayH/ tileH);

        if (format == RenderTileFormat::DISPLAY_RATIO) {
            float ratioDiffMin = numeric_limits<float>::max();
            float displayRatio = ((float) displayW) / ((float) displayH);

            for (int r = 1; r <= Rmax; r++) {
                if (numCam % r == 0 ) {
                    int c = numCam / r;
                    if (tileW * c < displayW) {
                        float ratio = (tileW * c) / (tileH * r);
                        if (abs(ratio - displayRatio) < ratioDiffMin) {
                            R = r;
                            C = c;
                            ratioDiffMin = abs(ratio - displayRatio);
                        }
                    }
                }
            }
        }
    }

    glm::mat4 toGlm(const misc::Pose& pose) {
        glm::mat4 Tcr = glm::mat4(1.0); // T_Wr_Cr
        Tcr = glm::translate(Tcr,glm::vec3(pose.poseMat.translation()(0),
                                           pose.poseMat.translation()(1),
                                           pose.poseMat.translation()(2)));
        Eigen::Vector3f rpy = pose.poseMat.rotation().eulerAngles(0,1,2);

        Tcr = glm::rotate(Tcr,rpy(0),glm::vec3(1,0,0));
        Tcr = glm::rotate(Tcr,rpy(1),glm::vec3(0,1,0));
        Tcr = glm::rotate(Tcr,rpy(2),glm::vec3(0,0,1));
        return Tcr;
    }



/**
 * set MP matrix of sceneShader except V s.t robotics ENU frame to opengl frame
 * ref: https://www.youtube.com/watch?v=x_Ph2cuEWrE
 * @param shader : soa.fs and soa.vs
 * @param FOV : FOV in degree
 * @param width: image width
 * @param height: image height
 */
    void initShaderTransform(Shader *shader, float FOV, int width, int height) {

        // Transformation from robotics ENU to opengl
        glm::mat4 world = glm::mat4(1.0); // T_Wr_Wg
        world = glm::rotate(world, float(3.141592/2.0), glm::vec3(1.0f, 0.0f, 0.0f));
        world = glm::rotate(world, -float(3.141592/2.0), glm::vec3(0.0f, 1.0f, 0.0f));
        shader->setMat4("world", glm::inverse(world));

        glm::mat4 model = glm::mat4(1.0); // just identity ?
        shader->setMat4("model", model);

        // Projection
        glm::mat4 projection = glm::perspective(glm::radians(FOV),
                                                (float) width / (float) height,
                                                0.0f, 500.0f);
        shader->setMat4("projection", projection);
    }

/**
 * set view matrix of shader with camPose (robotics world frame)
 * see (img/auto_chaser4 - Coordinate system.png)
 * @param shader : soa.fs and soa.vs
 * @param camPose : misc::Pose from robotics world frame (x-axis = optical)
 */
    void setCameraSingle(Shader *shader, const misc::Pose &camPose) {

        // Transformation from robotics ENU to opengl
        glm::mat4 world = glm::mat4(1.0); // T_Wr_Wg
        world = glm::rotate(world, float(3.141592/2.0), glm::vec3(1.0f, 0.0f, 0.0f));
        world = glm::rotate(world, -float(3.141592/2.0), glm::vec3(0.0f, 1.0f, 0.0f));

        // Camera pose in robotics frame
        glm::mat4 Tcr = toGlm(camPose);

        // Camera pose from robotics frame to opengl frame
        glm::mat4 Tc = glm::mat4(1.0); // T_Cg_Cr
        Tc = glm::rotate(Tc, float(3.141592/2.0), glm::vec3(1.0f, 0.0f, 0.0f));
        Tc = glm::rotate(Tc, -float(3.141592/2.0), glm::vec3(0.0f, 1.0f, 0.0f));

        // Camera pose in game frame
        glm::mat4 view = glm::mat4(1.0); // T_Cg_Wg
        view = glm::inverse(Tc) * glm::inverse(Tcr) * world;
        shader->setMat4("view", view);
    }

/**
 * read current buffer and save it to opencv mat
 * @param width
 * @param height
 * @return
 */
    cv::Mat readToCv(int width, int height) {
        // Render to cv matrix
        cv::Mat img(height, width, CV_8UC3);
//    glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3) ? 1 : 4);
//    glPixelStorei(GL_PACK_ROW_LENGTH, img.step/img.elemSize());
        glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, img.data);
        cv::flip(img, img, 0);
        return img;
    }

/**
 * render a single image w.r.t pose
 * @param shader
 * @param camPose
 * @param arg
 * @param tileRCMap
 * @return
 */
    cv::Mat render(Shader *shader, const misc::Pose &camPose,
                        RenderArgument arg, vector<cv::Rect>& tileRCMap ) {
        tileRCMap.clear();
        tileRCMap.push_back(cv::Rect(0,0,arg.tileWidth,arg.tileHeight));
        shader->use();
        initShaderTransform(shader, arg.renderFov, arg.tileWidth, arg.tileHeight);
        setCameraSingle(shader, camPose);
        glEnable(GL_DEPTH_TEST);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glBindVertexArray(
                arg.vao); // after binding VAO, every calls to VBO and glVertexAttribPointer is associated with this VAO
        glViewport(0, 0, arg.tileWidth, arg.tileHeight);
//        glDrawArrays(GL_TRIANGLES, 0, 3 * arg.nTriangle);
        glDrawElements(GL_TRIANGLES, 3 * arg.nTriangle, GL_UNSIGNED_INT,0);

        arg.tileRow = 1 ; arg.tileCol = 1;
        return readToCv(arg.tileWidth, arg.tileHeight);
    }

/**
 * render image tile (arg.tileRow x arg.tileCol) w.r.t poses in poseSet
 * @param shader scene shader
 * @param shaderTile tile shader
 * @param poseSet
 * @param arg
 * @param tileRCMap linear vector of cv::Rect for corresponding bound of poseSet
 * @return
 */
    cv::Mat renderTile(Shader *shader, Shader *shaderTile,
                            const PoseSet &poseSet, RenderArgument arg,vector<cv::Rect>& tileRCMap) {

        // setting scene shader
        initShaderTransform(shader, arg.renderFov,
                                 arg.tileWidth, arg.tileHeight);

        // 1. initialize tile vertices
        tileRCMap.clear();

        float* vertexTile = new float[arg.tileRow * arg.tileCol * 24];
        int cntTile = 0;

        // screen quad VAO
        float delX = 2.0 / arg.tileCol;
        float delY = 2.0 / arg.tileRow;

        float xmin, ymin, xmax, ymax;

        for (int r = 0; r < arg.tileRow; r++)
            for (int c = 0; c < arg.tileCol; c++) {
                xmin = -1.0 + c * delX;
                xmax = xmin + delX;
                ymax = 1.0 - r * delY;
                ymin = ymax - delY;

                float quadVertices[24] = { // vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
                        // positions   // texCoords
                        xmin, ymax, 0.0f, 1.0f,
                        xmin, ymin, 0.0f, 0.0f,
                        xmax, ymin, 1.0f, 0.0f,
                        xmin, ymax, 0.0f, 1.0f,
                        xmax, ymin, 1.0f, 0.0f,
                        xmax, ymax, 1.0f, 1.0f
                };

                for (int n = 0; n < 24; n++)
                    vertexTile[24 * cntTile + n] = quadVertices[n];

                cntTile++;
            }

        glBindBuffer(GL_ARRAY_BUFFER, arg.quadVbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexTile), &vertexTile, GL_STATIC_DRAW);


        glBindVertexArray(arg.vao);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear buffers to preset values from glClearColor
        glEnable(GL_DEPTH_TEST);


        // 2. draw on tiles
        unsigned int cnt = 0;
        for (int r = 0; r < arg.tileRow; r++)
            for (int c = 0; c < arg.tileCol; c++) {

//            tileRCMap.emplace_back(cv::Rect(c*arg.tileWidth, (arg.tileRow-r-1)*arg.tileHeight,
//                                            arg.tileWidth,arg.tileHeight));

                tileRCMap.emplace_back(cv::Rect(c*arg.tileWidth, r*arg.tileHeight,
                                                arg.tileWidth,arg.tileHeight));


                glBindFramebuffer(GL_FRAMEBUFFER, arg.frameBuffer);
                shader->use();
                // Draw triangles to framebuffer
                misc::Pose camPose = poseSet.get(cnt);
                setCameraSingle(shader, camPose);

                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear buffers to preset values from glClearColor
                glEnable(GL_DEPTH_TEST);
                glBindVertexArray(arg.vao);
                glViewport(0, 0, arg.tileWidth, arg.tileHeight);
//                shader->use();
//                glDrawArrays(GL_TRIANGLES, 0, 3 * (arg.nTriangle));
                glDrawElements(GL_TRIANGLES, 3 * arg.nTriangle, GL_UNSIGNED_INT,0);

                // now bind back to default framebuffer and draw a quad plane with the attached framebuffer color texture
                glBindFramebuffer(GL_FRAMEBUFFER, 0);
                glDisable(GL_DEPTH_TEST);
                shaderTile->use();
                glBindVertexArray(arg.quadVao);
                glBindTexture(GL_TEXTURE_2D,
                              arg.textureColorBuffer);       // use the color attachment texture as the texture of the quad plane
                glViewport(0, 0, arg.tileWidth * arg.tileCol, arg.tileHeight * arg.tileRow);
                glDrawArrays(GL_TRIANGLES, 6 * cnt, 6);
                cnt++;
            }

        // Render to cv matrix
        return readToCv(arg.tileWidth * arg.tileCol, arg.tileHeight * arg.tileRow);
    }

    void SceneRenderServer::insertVertex(float x, float y, float z, float r, float g, float b) {
        int startIdx = 6*(nVertex++);
        vertexPtr[startIdx] = x;
        vertexPtr[startIdx+1] = y;
        vertexPtr[startIdx+2] = z;
        vertexPtr[startIdx+3] = r;
        vertexPtr[startIdx+4] = g;
        vertexPtr[startIdx+5] = b;
    }

    void SceneRenderServer::insertIndex(unsigned int vertex1, unsigned int vertex2, unsigned int vertex3) {
        int startIdx = 3*(nTriIndex++);
        indexPtr[startIdx] = vertex1 ;
        indexPtr[startIdx+1] = vertex2 ;
        indexPtr[startIdx+2] = vertex3 ;
    }
    bool SceneRenderServer::initShader(string shaderRootDir) {

        // https://stackoverflow.com/questions/47918078/creating-opengl-structures-in-a-multithreaded-program
        /**
         * NOTE: This should be called in the same thread where rendering is requested
         */

        // create context
        if (! createOpenglContext()) {
            cerr << (myName + "E: opengl context was not created.") << endl;
            return false;
        }

        isShaderInit = true;
        // spawn buffer objects for scene rendering
        glGenVertexArrays(1,&openglObjs.VAO);
        glBindVertexArray(openglObjs.VAO);
        glGenBuffers(1,&openglObjs.VBO);
        glGenBuffers(1,&openglObjs.EBO);

        glBindBuffer(GL_ARRAY_BUFFER,openglObjs.VBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,openglObjs.EBO);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        // build and compile our shader program
        sceneShader = new Shader((shaderRootDir+"/soa.vs").c_str(), (shaderRootDir+"/soa.fs").c_str());
        sceneShader->use();
        tileShader = new Shader((shaderRootDir+"/framebuffers_screen.vs").c_str(), (shaderRootDir+"/framebuffers_screen.fs").c_str());
        tileShader->setInt("screenTexture", 0);
        // spawn buffer objects for tile rendering
        glGenVertexArrays(1, &openglObjs.quadVAO);
        glGenBuffers(1, &openglObjs.quadVBO);
        glBindVertexArray(openglObjs.quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, openglObjs.quadVBO);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

        // frame buffer setting
        glGenFramebuffers(1, &openglObjs.framebuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, openglObjs.framebuffer);
        glGenTextures(1, &openglObjs.textureColorBuffer);
        glBindTexture(GL_TEXTURE_2D, openglObjs.textureColorBuffer);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, param.imgWidth,param.imgHeight,
                     0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, openglObjs.textureColorBuffer, 0);

        glGenRenderbuffers(1, &openglObjs.rbo);
        glBindRenderbuffer(GL_RENDERBUFFER, openglObjs.rbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8,
                              param.imgWidth,param.imgHeight); // use a single renderbuffer object for both a depth AND stencil buffer.
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER,openglObjs.rbo); // now actually attach it

        return true;
    }

    bool SceneRenderServer::createOpenglContext() {

        // memory allocation
        vertexPtr = new float [nMaxTriangles*18];
        indexPtr = new unsigned int [nMaxTriangles*3];

        // opengl context creation
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // do not open window

        // size of window should be larger than total tile size
        GLFWwindow* window = glfwCreateWindow(param.displayWidth,
                                              param.displayHeight,
                                              "iswy", NULL, NULL);
        if (window == NULL)
        {
            std::cerr<< "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return false;
        }
        glfwMakeContextCurrent(window);

        // AFTER context creation, GLEW initialization
        glewExperimental = GL_TRUE;
        GLenum errorCode = glewInit();
        if (GLEW_OK != errorCode) {
            std::cerr << "Failed to initialize GLEW " << glewGetErrorString(errorCode) << std::endl;
            glfwTerminate();
            exit(EXIT_FAILURE);
            return false;
        }

        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);


        return true;
    }

/**
 * upload vertices for scene render.
 */
    void SceneRenderServer::uploadVertexToRender() {

        if (getNumVertex() ==0 || getNumIndex() ==0){
            printStr("E: trying bind with zero-sized vertex array." );
        }

        // upload vertices of static obstacle to the buffer of manager
        glBindVertexArray(openglObjs.VAO);
        glBindBuffer(GL_ARRAY_BUFFER,openglObjs.VBO);
        glBufferData(GL_ARRAY_BUFFER,sizeof(float)*getNumVertex()*6,vertexPtr,GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,openglObjs.EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     sizeof(unsigned int)*(getNumIndex()*3),indexPtr,GL_DYNAMIC_DRAW);
        printStr("uploaded vertex to buffer for future render request.");
    }

/**
 * Read soa mesh and non-soa mesh, bind vertex and index for rendering
 * @param soaMeshQeue
 * @param nonSoaMeshQueue
 * @return
 */
    bool SceneRenderServer::uploadMesh(const o3d_tensor::TriangleMesh& mesh) {

        // Refresh vertices Array from mesh
        nVertex =0; nTriIndex = 0; // vertex number will increase as insert vertex is performed
        int nTriangles = mesh.GetTriangleIndices().NumElements()/3;
        int nVertices = mesh.GetVertexPositions().NumElements()/3;

        // vertex + COLOR
        auto& tVert = mesh.GetVertexPositions();
        auto curVertexPtr = (float *)tVert.GetDataPtr();
        auto& tCol = mesh.GetVertexColors();
        auto curColorPtr = (float *)tCol.GetDataPtr();
        for (int i = 0 ; i < nVertices; i++){
            float x,y,z,r,g,b;
            x = curVertexPtr[3*i];
            y = curVertexPtr[3*i+1];
            z = curVertexPtr[3*i+2];
            r = curColorPtr[3*i] ;
            g = curColorPtr[3*i+1] ;
            b = curColorPtr[3*i+2] ;

//            printf("X = [%.2f , %.2f , %.2f ] / C = [%.2f , %.2f , %.2f ]\n",x,y,z,r,g,b);
            insertVertex(x,y,z,r,g,b);
        }

        // index
        auto& tIndex = mesh.GetTriangleIndices();
        auto tIndexCont  = tIndex.To(o3d_core::Int32).Contiguous();
        auto curIndexPtr = (unsigned int *)tIndexCont.GetDataPtr();
        for (int i = 0 ; i < nTriangles; i++){
            unsigned int v1,v2,v3;
            v1 = curIndexPtr[3*i];
            v2 = curIndexPtr[3*i+1];
            v3 = curIndexPtr[3*i+2];
//            printf("[%d , %d , %d ]\n",v1,v2,v3);
            insertIndex(v1,v2,v3);
        }

        printStr("Current triangle=" + to_string(nTriangles) +" / vertices=" + to_string(nVertex));

        // 2. Bind buffer for future rendering
        uploadVertexToRender();
        return true;
    }


    RenderResult SceneRenderServer::renderService(const PoseSet &camPoseSet, const o3d_tensor::TriangleMesh& mesh) {


        Timer timer;
        // Upload mesh to opengl buffer
        RenderResult renderResult;
        if (! isShaderInit){
            printf("shader has not yet initialized. Does not shade.\n");
            return renderResult;
        }

        if (! uploadMesh(mesh)){
            renderResult.isRenderSuccess = false;
            return renderResult;
        }

        RenderArgument arg;
        arg.nTriangle = nMesh();
        arg.tileHeight = param.imgHeight;
        arg.tileWidth = param.imgWidth;
        arg.renderFov = param.fovDeg;

        // tile arrangement?
        arrangeTile(RenderTileFormat::DISPLAY_RATIO,
                    arg.tileWidth, arg.tileHeight,
                    param.displayWidth,
                    param.displayHeight,
                    camPoseSet.size(),
                    arg.tileRow,arg.tileCol);
        renderResult.tileRow = arg.tileRow;
        renderResult.tileCol = arg.tileCol;
        renderResult.viewPoseSet = camPoseSet;

        arg.vao = openglObjs.VAO;
        arg.quadVao = openglObjs.quadVAO;
        arg.quadVbo = openglObjs.quadVBO;  // in case of tile, we will feed a set of vertices
        arg.frameBuffer = openglObjs.framebuffer;
        arg.textureColorBuffer = openglObjs.textureColorBuffer;

        // rendering
        renderResult.renderImage = renderTile(sceneShader,tileShader,camPoseSet,arg,renderResult.tileMap);
        double elapse = timer.stop();
        string message = "Evaluator: total render process took " + to_string(elapse) + " ms";
        printStr(message);
        renderResult.elapse = elapse;
        renderResult.isRenderSuccess = true;
        return renderResult;
    }
};