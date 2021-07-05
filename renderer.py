import pybullet as p
import time
import pybullet_data
import random
from PIL import Image
import os, glob
import math
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation


def renderGenerator(mode, colors, allurdfWithClassId, rng, nbobj, nbview, outFolder, sceneName):
    p.connect(mode)
    # reset la simulation
    p.resetSimulation()

    #reduce timeStep if the physics become unstable
    #but you will need to take more steps before the system reach equilibrium
    timeStep = 0.01
    p.setTimeStep(timeStep)

    # PyBullet_data package
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # chargement du fichier plane.urdf (URDF = Unified Robot Description Format)
    planeId = p.loadURDF("plane.urdf")

    objs = [planeId]

    meshScale = [0.1,0.1,0.1]

    collisionDict = {}

    segIdToClass= [0]

    for i in range(nbobj):
      cubeStartPos = [rng.uniform(-30,30),rng.uniform(-30,30),rng.uniform(10,50)]
      cubeStartOrientation = p.getQuaternionFromEuler([rng.uniform(-math.pi,math.pi),rng.uniform(-math.pi,math.pi),rng.uniform(-math.pi,math.pi)])
      try:
        print("loading object " + str(i) + " / " + str(nbobj) )
        (objname,objid) = rng.choice(allurdfWithClassId)
        print(objname)
        #we don't use loadURDF to be able to change the color of the bricks easily
        #objs.append( p.loadURDF(objname,cubeStartPos,cubeStartOrientation ))

        #When we create multiple instances of the same object
        #there are some performance gain to share the shapes between rigid bodies
        #Instead of p.loadURDF you can use the following
        name = Path(objname).name
        nameroot = name[:-5]
        partition = objname.split("/")[0]
        color = rng.choice(colors)
        vs = p.createVisualShape(p.GEOM_MESH,fileName=partition+"/"+nameroot+"/"+nameroot+".obj",meshScale=meshScale, rgbaColor=color )
        #because of the color is inside the visual shape we don't share the visual shape so similar instances can have different colors
        cuid = 0
        #but we share the collision shapes
        if name in collisionDict:
            cuid = collisionDict[name]
        else:
            cuid = p.createCollisionShape(p.GEOM_MESH,fileName=partition+"/"+nameroot+"/"+nameroot+"_vhacd.obj",meshScale=meshScale)
            collisionDict[name]=cuid
        objs.append( p.createMultiBody(1.0,cuid,vs,cubeStartPos,cubeStartOrientation) )
        segIdToClass.append(objid)
      except:
        print("failed to load : " + objname )

    p.setGravity(0,0,-10)

    nbsteps = 500
    for i in range(nbsteps):
      p.stepSimulation()
      print("step " + str(i) + " / " + str( nbsteps ) )

    for j in range(nbview):
        print("rendering view " + str(j))
        ang = rng.uniform(-math.pi,math.pi)
        r = rng.uniform(1,30)
        h = rng.uniform(50,100)
        fov = 45
        aspect = 1
        nearVal = 0.1
        farVal = 150.1

        camParams = [r*math.cos(ang),r*math.sin(ang),h,   0,0,0,   0,0,1, fov,aspect,nearVal,farVal ]

        print("camParams")
        print(camParams)


        viewMatrix = p.computeViewMatrix(
            cameraEyePosition=camParams[0:3],
            cameraTargetPosition=camParams[3:6],
            cameraUpVector=camParams[6:9])

        projectionMatrix = p.computeProjectionMatrixFOV(
            fov=camParams[9],
            aspect=camParams[10],
            nearVal=camParams[11],
            farVal=camParams[12])

        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=1080,
            height=1080,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix)

        Path(outFolder).mkdir(parents=True, exist_ok=True)
        viewStr = "-view"+str(j)
        im = Image.fromarray(rgbImg)
        im.save(outFolder +"/" + sceneName + viewStr + ".png")

        #We also compute here information which will be interesting to try to predict
        #Typically we are going to ask a neural network to predict
        # or each pixel
        #the offset to the centroid/center of the object the pixel belongs to
        #For each pixel the NN is making a prediction of which class it belongs to
        #For each pixel the NN is making a prediction of the orientation of the object belongs to
        #For each pixel the NN is making a prediction of the 3D bounding box coordinates the object belongs to
        #...
        #Then all you need is to run a clustering algorithm like DBSCAN or HDBSCAN on the neural network prediction

        #In pixel space the 2D centroid of all the visible pixels of the object
        #if no pixel from the object is visible in the image the row will contain nan
        centroids = np.array( [ np.mean( np.argwhere(segImg == id ),axis=0) for id in range(len(objs))  ])

        #In absolute world coordinates
        positionAndOrientation = [ p.getBasePositionAndOrientation(id) for id in objs]
        pos = np.array( [x[0] for x in positionAndOrientation] )
        orient = np.array( [x[1] for x in positionAndOrientation ] )

        #We compute the coordinates of the center of the object in screen Coordinates
        pM = np.reshape(projectionMatrix, (4, 4))
        vM = np.reshape(viewMatrix, (4, 4))
        pos1 = np.hstack([pos,np.ones( (np.shape(pos)[0],1))])
        xyzw = np.dot(np.dot(pos1,vM),pM)
        posInNDC = xyzw[:,0:3] / xyzw[:,3:]
        viewSize = np.reshape( np.array([width,height],dtype=np.float32),(1,2))
        viewOffset = viewSize / 2.0 -0.5
        posInScreenCoordinates = posInNDC[:,0:2] * viewSize + viewOffset
        depthInScreenCoordintes = posInNDC[:,2]

        #We can do the same technique for each corner of the 3d bounding box of the object
        #...

        #For the orientation of the object in view coordinates
        rv = Rotation.from_matrix(vM[0:3,0:3])
        orientationInViewCoordinates = np.array([ (rv * Rotation.from_quat( orient[i] ) ).as_quat() for i in range(orient.shape[0])])

        # We don't allow_pickle for safety reasons
        np.savez(outFolder + "/" + "parameters_" + sceneName + viewStr,
                 camParams=np.array(camParams),
                 position=pos,
                 orientation=orient,
                 centroids=centroids,
                 posInScreenCoordinates=posInScreenCoordinates,
                 orientationInViewCoordinates = orientationInViewCoordinates,
                 depthInScreenCoordintes= depthInScreenCoordintes,
                 segImg= segImg,
                 segIdToClass = np.array(segIdToClass),
                 allow_pickle=False)

        #don't forget to put the allow_pickle=False in the load (that's where it matters!)
        #with np.load(outFolder + "/" + "parameters" + sceneName + viewStr + ".npz",allow_pickle=False ) as data:
        #    print("npz keys :")
        #    for k in data.files:
        #        print(k)
        #        print(data[k])

        #We don't store output rasterized images as they will be generate on the fly from the segmentation map
        #With something like the following
        #classImg = np.array(segIdToClass)[ segImg.reshape((-1,))].reshape(segImg.shape)
        #The offsets in pixel space will have to be computed on the fly to from the absolute coordinates in pixel space


    #instead of stepping manually you can step in real time,
    #but be careful that you don't have too many objects or it may become unstable
    #p.setRealTimeSimulation(1)

    if mode == p.GUI:
      while True:
        time.sleep(0.01)
        p.stepSimulation()

    p.disconnect()

if __name__=="__main__":
    # use mode = p.DIRECT for usage without gui
    #mode = p.GUI
    mode = p.DIRECT
    # In GUI mode it became slow at around 500 objects
    # In Direct mode, I could run it OK with 3000 similar object
    nbColors = 3

    rng = random.Random(42)
    colors = [[rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1), 1.0] for i in range(nbColors)]

    # we exclude all files beginning by _ for example _prototype.urdf
    allurdf = sorted(glob.glob("lego-*/[!_]*.urdf"))
    print("Number of urdf :" + str(len(allurdf)))

    nbobj = 100

    nbview = 10

    #We reserve class 0 for background
    allurdfWithClassId = [ (allurdf[i],i+1) for i in range(len(allurdf))]
    rng = random.Random(42)
    renderGenerator(mode, colors,allurdfWithClassId, rng,nbobj,nbview,"Renderings", "Scene0")
    #It 's deterministic and we get the same results
    rng = random.Random(42)
    renderGenerator(mode, colors, allurdfWithClassId, rng,nbobj,nbview,"Renderings", "Scene0bis")
