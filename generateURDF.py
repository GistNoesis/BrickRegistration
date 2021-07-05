import os
import sys
import threading
from object2urdf import ObjectUrdfBuilder
import shutil

# Build single URDFs
object_folder = "lego"

#An ugly copy paste of build_library to catch exception
def safe_build_library(urdfbuilder, **kwargs):
        print("\nFOLDER: %s"%(urdfbuilder.object_folder))

        # Get all OBJ files
        obj_files  = urdfbuilder._get_files_recursively(urdfbuilder.object_folder, filter_extension='.obj', exclude_suffix=urdfbuilder.suffix)
        stl_files  = urdfbuilder._get_files_recursively(urdfbuilder.object_folder, filter_extension='.stl', exclude_suffix=urdfbuilder.suffix)       

        obj_folders=[]
        for root, _, full_file in obj_files:
            obj_folders.append(root)
            try:
              urdfbuilder.build_urdf(full_file,**kwargs)
            except:
              print("An exception occured during " + full_file )
            common = os.path.commonprefix([urdfbuilder.object_folder,full_file])
            rel = os.path.join(full_file.replace(common,''))
            print('\tBuilding: %s'%(rel) )
        
        for root, _, full_file in stl_files:
            if root not in obj_folders:
                try:
                  urdfbuilder.build_urdf(full_file,**kwargs)
                except:
                  print("An exception occured during " + full_file)
                
                common = os.path.commonprefix([urdfbuilder.object_folder,full_file])
                rel = os.path.join(full_file.replace(common,''))
                print('Building: %s'%(rel) )

def thread_function(tindex,nthreads):
  shutil.copy( "_prototype.urdf",object_folder+"-"+str(tindex),follow_symlinks=True)
  builder = ObjectUrdfBuilder(object_folder+"-"+str(tindex))
  #we use center = "geometry" instead of "mass" because it fails on some objects and make the program crash
  #we use depth=1 as an extra parameter for vhacd so that it sacrifice collision geometry quality so that 
  #it goes faster during simulation
  #oclAcceleration=0
  safe_build_library(builder,force_overwrite=True, decompose_concave=True, force_decompose=False, center = 'geometry',depth=1)
  
  
nthreads = 8
for i in range(nthreads):
  x = threading.Thread(target=thread_function, args=(i,nthreads))
  x.start()