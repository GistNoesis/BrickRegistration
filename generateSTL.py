import os, glob
from pathlib import Path
import subprocess
import threading

allfiles = sorted(glob.glob( "ldraw/parts/*.dat"))

def thread_function(tindex,nthreads):
  ind = tindex
  while ind < len(allfiles):
    path = allfiles[ind]
    name = Path(path).name
    print(name)
    nameroot= name[:-4]
    p = "lego-" + str(tindex) +"/"+nameroot
    Path( p ).mkdir(parents=True,exist_ok=True)
    with open(p+"/"+nameroot+str(".stl"),"w") as outfile:
      subprocess.run(["bin/dat2stl", "--file", path, "--ldrawdir", "./ldraw"],stdout=outfile)
    ind = ind +nthreads
    print(str(ind) + "/" + str(len(allfiles)))


nthreads = 8
for i in range(nthreads):
  x = threading.Thread(target=thread_function, args=(i,nthreads))
  x.start()
