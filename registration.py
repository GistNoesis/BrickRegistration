import tensorflow as tf
import glob
import numpy as np
from pathlib import Path
from PIL import Image
from tensorflow.keras import layers
import hdbscan

globalrng = np.random.default_rng(42)
#Passing it as global variable because tf.data.Dataset.from_generator complain when passed as argument


def generateSegmentationSamples(folder, size, nbclass, nbWrongCandidates,outputInCandidateSpace):
    allfiles = sorted(glob.glob(folder.decode('utf-8')+"/*.png"))
    #we do the shuffling, at each epoch the globalrng state is different
    perm = globalrng.permutation(len(allfiles))

    for i in range(len(allfiles)):
        fullpath = allfiles[perm[i]]
        path = Path(fullpath)
        name =path.name[:-4]
        dirname = str(path.parent)
        outputname = dirname+"/parameters_"+name+".npz"

        img = np.array(Image.open(fullpath).convert("RGB"))
        shp = img.shape
        xmin = globalrng.integers(0,shp[0]-size[0])
        ymin = globalrng.integers(0,shp[1]-size[1])

        # don't forget to put the allow_pickle=False in the load (that's where it matters!)
        with np.load(outputname,allow_pickle=False ) as data:
            segImg = data["segImg"][xmin:xmin+size[0],ymin:ymin+size[1]]
            #the unique should be done at the batch level but we can't because of the custom_loss bug
            #Because of this it will only work when bach_size = 1
            wrongCandidates = globalrng.integers(0, nbclass,nbWrongCandidates )
            candidateIds =  np.unique( np.concatenate( [data["segIdToClass"],wrongCandidates]))
            mapObjId = np.reshape(data["segIdToClass"][np.reshape(segImg, -1)], segImg.shape)

        img = img[xmin:xmin+size[0],ymin:ymin+size[1],:]

        if outputInCandidateSpace == True:
            candMap = np.zeros((nbclass), )
            candMap[candidateIds] = np.arange(candidateIds.shape[0])

            remapped = np.reshape(candMap[np.reshape(mapObjId, -1)], mapObjId.shape)
            targetImg = remapped
        else:
            targetImg=mapObjId

        yield (img,candidateIds), targetImg
        #epoch = epoch+1


class TableDistance(layers.Layer):
    def __init__(self,tableSize, **kwargs):
        self.tableSize = tableSize
        super(TableDistance, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dims = len(input_shape)
        shape = [1]*(len(input_shape)-1) + [input_shape[-1],self.tableSize]
        self.distanceScale = self.add_weight(name="scale",shape=[1]*(self.input_dims),initializer=tf.constant_initializer(1.0))
        self.table = self.add_weight(name='table', shape=shape,
                                      initializer='glorot_uniform', trainable=True
)
        super(TableDistance, self).build(input_shape)

    def call(self, input):
        #This can be optimized
        out = tf.abs(self.distanceScale)* tf.reduce_sum( tf.square(self.table - tf.expand_dims( input,axis=-1)),axis=-2)
        return out

    def compute_output_shape(self, input_shape):
        print("output_shape")
        output_shape = input_shape[:-1]  + (self.tableSize,)
        print(output_shape)
        return output_shape

#This is a layer to tackle the problem of high class count
#We ask to predict the class among a subset of candidates
#So we extract from the full kernel only relevant rows
#It must be use either with a custom loss to do the remapping or by providing y_true indexed in candidates space
#the candidates are repeated along the batch dimension if bs!=1 by the dataset batching
#so the intended candidates should be candidates[0]
#No activation Loss so that it ouputs logits
class SelectableConv2D(layers.Layer):
    def __init__(self,totalfilters,kernelSize, **kwargs):
        self.totalfilters = totalfilters
        self.kernel_size = (kernelSize, kernelSize)
        super(SelectableConv2D, self).__init__(**kwargs)

    def build(self, input_shapes):
        shape = (self.totalfilters, self.kernel_size[0],self.kernel_size[1] ,input_shapes[0][-1] )
        self.kernel = self.add_weight(name='kernel', shape=shape,
                                      initializer='glorot_uniform',trainable=True)
        super(SelectableConv2D, self).build(input_shapes)

    def call(self, inputs):
        candidates = inputs[1]
        subk = tf.gather_nd( self.kernel,tf.reshape(candidates[0],(-1,1)) )
        subkernel = tf.transpose( subk,perm=[1,2,3,0] )
        return tf.nn.conv2d(inputs[0], subkernel,strides=[1,1,1,1],padding="SAME")

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]  + (-1,)


def buildModel( maxNumberOfClasses ,outputInCandidateSpace):
    #we use some U-net and stich the pieces together
    inp = tf.keras.Input(shape=(None, None, 3))
    candidateIds = tf.keras.Input(shape=(None,), dtype=tf.int32)
    img = tf.keras.layers.experimental.preprocessing.Rescaling(scale = 1./255)(inp)
    #We build a simple network here
    #Feel free to do some UNet segmentation
    #more layers, resnets
    conv = tf.keras.layers.Conv2D( 100, 5, padding="SAME",activation="selu")(img)

    #conv = tf.keras.layers.Conv2D(3, 5, padding="SAME", activation="selu")(img)
    #If we have a low number of classes we can use the standard
    #conv = tf.keras.layers.Conv2D(maxNumberOfClasses, 5, padding="SAME", activation=None)(conv)
    #If we have a high number of classes there are various tricks we can use to reduce the memory usage and increase computation speed
    if outputInCandidateSpace == False:
        #The easiest is to reduce the number of features in the layer just above (aka low dimension embedding)
        key = tf.keras.layers.Conv2D(3, 5, padding="SAME", activation=None)(img)
        out = tf.keras.layers.Conv2D(maxNumberOfClasses, 5, padding="SAME", activation=None)(key)
        #alternatively you can use a near-neighbor for prediction
        #out = TableDistance(maxNumberOfClasses)(key)
        #although the above doesn't offer performance benefits you can write a sparse version KNNDistance where old-school techniques
        #like indexing apply to be able to handle huge number of class
        #But remember that near neighbor training are slower to converge as they are only modified when falling in the neighborhood
    else:
        #For example we can subselect the candidates among a pool of candidates
        out = SelectableConv2D(maxNumberOfClasses,5)([conv,candidateIds])

    #Alternatively we can output multiple ouputs the first output corresponding to the high order bits of the classId,
    #and the second output to the low order bits of the classId
    #this can work well when the classId are in an observable "sorted" order for example from smaller piece to bigger pieces

    #Alternatively using multiple maps you can reinvent Ferns
    #Alternatively if you use the prediction of the high order bits as an input to predict (or subselect) of the low order bits you can reinvent Trees
    #Alternatively you can use transformers or a variant of LSH

    #Alternatively you can learn independently object caracteristics instead of class_id and predict those and identify the class_id in post_processing

    #You can also add multiple output like predicting the centroid,..., and the content of the npz archive
    #It will help the clustering algorithm to distingate similar object
    model = tf.keras.Model(inputs=[inp,candidateIds], outputs=[out], name="lego_segment")

    '''
    #Not working due to issue #47311
    def custom_loss(candidateIds):
        # Create a loss function that remaps the id of the class to the id inside the candidate list
        def loss(y_true, y_pred):
            uniqIds = tf.unique(candidateIds)
            indices = tf.reshape(uniqIds, (-1, 1))
            updates = tf.range(tf.shape(indices)[0])
            shape = tf.constant([maxNumberOfClasses])
            candMap = tf.scatter_nd(indices, updates, shape)
            #The following won't work when batch_size != 1
            y_true_remapped = tf.reshape(tf.gather(candMap, tf.reshape(y_true, (-1,))), tf.shape(y_true))
            #when bs > 1 something like this should do the trick
            #y_trueobj = [ (candidateIds[j][y_true[j].reshape((-1,) ]).reshape(y_true[j].shape) for j in range(bs) ]
            #y_true_remapped = candMap[y_trueobj]
            return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true_remapped,y_pred)
        # Return a function
        return loss
    '''
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    return model

def cluster( img ):
    xi = np.zeros_like(img)+np.reshape( np.arange(img.shape[1],dtype=np.int32) ,(1,img.shape[1],1,1) )
    yi = np.zeros_like(img)+np.reshape( np.arange(img.shape[2],dtype=np.int32) ,(1,1,img.shape[2],1) )

    features = np.concatenate([img*1000,xi,yi],axis=-1)

    flatfeatures = np.reshape(features,(-1,3))
    flatimg = np.reshape(img,(-1))
    nz = np.reshape( np.argwhere( flatimg != 0 ),(-1,))
    nzfeatures = flatfeatures[nz]

    if nzfeatures.shape[0] == 0:
        return np.zeros((0,),dtype=np.int32),nzfeatures
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(nzfeatures)
    return clusterer.labels_,nzfeatures

def demoCluster():
    subImageSize = (100, 100)
    img = np.zeros( (1,)+ subImageSize +(1,) , dtype=np.int32)
    img[0,66:85,75:80,0] = 4
    img[0, 33:56,45:55,0] = 6
    img[0, 88:96, 45:55, 0] = 6
    img[0, 0, 0, 0] = 1
    print( cluster(img) )

def demo():
    maxNumberOfClass = 15000
    numberOfWrongExtraId = 100
    subImageSize = (100,100)
    outputInCandidateSpace = True
    folder = "Renderings"
    ds =tf.data.Dataset.from_generator(generateSegmentationSamples, args=[folder,subImageSize,maxNumberOfClass,numberOfWrongExtraId,outputInCandidateSpace],
                                       output_types=( (tf.uint8,tf.int32),tf.int32),
                                       output_shapes=(((None,None,3),(None,)),(None,None)) )
    model = buildModel( maxNumberOfClass , outputInCandidateSpace)

    model.fit( ds.batch(1), epochs=10)

    #We iterate on sub images from the dataset to test that everything runs
    #If you have enough memory you can run on the full image and all candidate classes in one go
    #otherwise you will have to do some image stiching
    #To do the stiching correctly you should use "VALID" padding convolution and predict a smaller output image than the input
    #When layers have dilations you should take care to align them properly
    #Look for UNet segmentation on the web for how to
    for batch in ds.batch(1).take(100):
        #print(batch)
        print("batch:")
        #print(batch)
        res = model(batch[0])
        id = tf.expand_dims(tf.argmax(res,axis=-1),axis=-1).numpy()
        labels,nzfeat = cluster(id)
        print("nbdistinct objects : ")
        if labels.shape[0] >0 :
            print(labels.max())
        else:
            print(0)



if __name__=="__main__":
    demo()
