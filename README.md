# Project for AI cameraman 
___

## Executables 
### RunOpen3dOpengl


### RenderVsReal
![real vs render](https://user-images.githubusercontent.com/30062474/152312829-703b4903-834a-498f-9647-f2d32c0bd05c.PNG)
This run file compares the real image view vs rendered view from the mesh (accumulated from .svo file). 
real image view 
When running this, a comparison view between the two will pop-up.  
#### 1. Required data 
A [dataset directory](https://mysnu-my.sharepoint.com/:f:/g/personal/a4tiv_seoul_ac_kr/Eil7djHq3ENAg4bxq2YPqhEBKL2pLj95TX-B_mn1ksiXQw?e=aDF5yL) should contain the followings.
* `calibration_result.txt`: a single row is composed of {edelkrone state (slider,pan,tilt), zed pose (1x16 flattened pose) at a recording sequence (`<=nCam`). 
* `image_n.png` : an image capture at each recording sequence (`<=nCam`). 
* `record.svo` : entire camera data history. 

#### Initialization
* Still, I could not make argument parsing part. Just modify `nCam`, `userName` and `datasetDir` in the source code. 
* Also, to make up the most similar look of the real world, we have to tune the opengl view matrix. 
