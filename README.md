# BrickRegistration

![Screenshot](https://raw.githubusercontent.com/GistNoesis/BrickRegistration/main/DemoRenderings/Scene0-view0.png)

This project is a tool to generate synthetic 3d scenes for toying with bricks registration.

The 3d models come from https://www.ldraw.org/library/updates/complete.zip where they are available with a permissive licence

This file is extracted into the repository 

I convert the model from ldraw format to stl :

`python3 generateSTL.py` which under the hood call that is embedded https://github.com/kristov/ldraw2stl

Then I convert the STL to the URDF :
`python3 generateURDF.py` which under the hood use https://pypi.org/project/object2urdf/ (`pip install object2urdf` )

These two commands take ~30min and 6hours to run.
The generateURDF use GPU and crash on some files if there is not enough memory (GTX 1080 Ti is OK)

It successfully generates ~14000 urdf files provided that your machine is beefy enough.
I have create a compressed 1.4GB legoSTLandURDF.tar.gz archive with them http://orchid.gistnoesis.net/legoSTLandURDF.tar.gz

You can download it here and extract it in the repository instead of having to run the above commands

Once you have generated the urdfs you can run :
`python3 renderer.py`

It will produce some renderings in a folder called "Renderings", and useful segmentation information.

You can edit the code and use GUI mode, if you just want to play with pybullet

Once you have generated a few renderings you can run :
`python3 registration.py`

It will create a neural network, train it, and run it and do some clustering to identify the bricks.
It is intended as a boiler plate to experiment with object recognition to get you started :)
Only a mock network, on a mock dataset have currently been run so some bugs are probably hiding.

More information is available inside the source files. Look at them.

If you have an iphone you can try similar technology from https://brickit.app/ 

This is a week-end project trying to replicate a technology ( https://news.ycombinator.com/item?id=27693560 )

Dependencies (pick the ones you need depending on what you want to run): 

pybullet tensorflow2 scipy object2urdf hdbscan 


