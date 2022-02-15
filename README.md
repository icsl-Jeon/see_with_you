# Project for AI cameraman 
___

## Installtion 
Due to ZED SDK, Windows 10 cannot build this library in Debug. The possible builds are RelWithDebInfo and Release. 
If we want to debug our library, we have to build this in RelWithDebInfo. The regarding issue is well explained [here](https://github.com/google/googletest/tree/main/googletest#incorporating-into-an-existing-cmake-project) also.

### Dependencies (build and install the belows)
* CUDA 11.3 
* OPENCV 4.5.4
* [yaml-cpp](https://github.com/jbeder/yaml-cpp): This should be built in RelWithDebInfo to cater both Release and RelWithDebInfo. 
Found that if this is built in Debug, this `LoadFile` function works only when in Debug mode, which is **not possible** for our case.
Do not forget to include PATH variables: `C:\Program Files (x86)\YAML_CPP\lib` and `C:\Program Files (x86)\YAML_CPP\bin`.
* [google-test](https://github.com/google/googletest/): We accept the recommendation of developers of Google. Download the master and put it in `./googletest`.  

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