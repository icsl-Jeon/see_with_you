# Project for AI cameraman 
___

## Installtion 
Due to ZED SDK, Windows 10 cannot build this library in Debug. The possible builds are RelWithDebInfo and Release. 
If we want to debug our library, we have to build this in RelWithDebInfo. The regarding issue is well explained [here](https://github.com/google/googletest/tree/main/googletest#incorporating-into-an-existing-cmake-project) also.

### Dependencies (build and install the belows)
* ZED SDK 3.6.5: This installs its own dependencies and registers them into `PATH`. 
```
C:\Program Files (x86)\ZED SDK\dependencies\freeglut_2.8\x64
C:\Program Files (x86)\ZED SDK\dependencies\glew-1.12.0\x64
C:\Program Files (x86)\ZED SDK\dependencies\opencv_3.1.0\x64
```
Thus, we have to ensure the version conflicts.  
* CUDA 11.3 
* OPENCV 4.5.4: We have to provide directory including `OpenCVConfig.cmake` and `OpenCVConfig-version` toward `PATH`. This was `C:\Users\junbs\OneDrive\Documents\window_dev\opencv\build` for my case. 
Ensure to build CUDA_MODULES.
* [yaml-cpp](https://github.com/jbeder/yaml-cpp): This should be built in RelWithDebInfo to cater both Release and RelWithDebInfo. 
Found that if this is built in Debug, this `LoadFile` function works only when in Debug mode, which is **not possible** for our case. Ensure to turn on `BUILD_SHARED_LIBS=TRUE`
Do not forget to include PATH variables: `C:\Program Files (x86)\YAML_CPP\bin` (for dynamic linking).
* [google-test](https://github.com/google/googletest/): We accept the recommendation of developers of Google. Download the master and put it in `./googletest`.  
* [Open3d](https://github.com/icsl-Jeon/Open3D):  For Windows, this was built in RelWithDebInfo and installed into 'C:\Program Files (x86)\Open3D'. 
Add `C:\Program Files (x86)\Open3D` into `PATH` variable. I use version 0.15.1 and it has no error regarding [tbb_static](https://github.com/icsl-Jeon/window_dev#open3d-tbb_staticlib-issues). I could run code out-of-the-box.

## Executables 
Recommend run in Release mode. RelWithDebInfo seems to stop with an unknown issue. 

### 1. RunOpen3dOpengl


### 2. RenderVsReal
![real vs render](https://user-images.githubusercontent.com/30062474/152312829-703b4903-834a-498f-9647-f2d32c0bd05c.PNG)
This run file compares the real image view vs rendered view from the mesh (accumulated from .svo file). 
real image view 
When running this, a comparison view between the two will pop-up.  

####  Required data 
A [dataset directory](https://mysnu-my.sharepoint.com/:f:/g/personal/a4tiv_seoul_ac_kr/Eil7djHq3ENAg4bxq2YPqhEBKL2pLj95TX-B_mn1ksiXQw?e=aDF5yL) should contain the followings.
* `calibration_result.txt`: a single row is composed of {edelkrone state (slider,pan,tilt), zed pose (1x16 flattened pose) at a recording sequence (`<=nCam`). 
* `image_n.png` : an image capture at each recording sequence (`<=nCam`). 
* `record.svo` : entire camera data history. 

#### Initialization
* Still, I could not make argument parsing part. Just modify `nCam`, `userName` and `datasetDir` in the source code. 
* Also, to make up the most similar look of the real world, we have to tune the opengl view matrix. Still, the current setting was found to be the best 


### 3. ViewScoring  
This executable senses the shooting scene where actor and OOI are being tracked. Then, given the scene interpretation, 
we scores each candidate view for better shooting angle.

#### Required data  