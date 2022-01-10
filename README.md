# Monocular Visual Odometry   

* * *    

## Explanation for this repository   
it is monocular visual odometry for KITTI dataset in Python.    
reference repository : <https://github.com/uoip/monoVO-python>   

original data is more than 4000 frame but it is so huge for git hub repository.   
therefore data in this repository is 200 frame images so playtime is very short.   
if you have original data, you can use it by editting path in "test.py"   

**test.py, visual_odometry.py** --> it is almost same code in reference repository   
**grid feature.py** --> it is code changed method that find visual feature

**Monocular_Visual_Odometry/report**  -->  you can read report in korean   

* * *   

## Comparison in algorithm
there are comparison about result between FAST argorithm and Grid feature   

<img src="image material/map_21.196328m.png" width="40%" height="30%"></img>
<img src="image material/map_grid&filter_40step_7.389475m.png" width="40%" height="30%"></img>   
### FAST argorithm(Mean Error=21.20m)　　　Grid feature(Mean Error=7.39m)   
  
* * *   
 
## Demonstration video
in this video, i use original data that is 4000 frame and upload video quickly played

<img src="image material/GOMCAM-20211202_1609280168.gif" width="80%" height="40%"></img>
