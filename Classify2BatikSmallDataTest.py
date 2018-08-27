from keras.models import load_model
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os, shutil
from keras.preprocessing.image import img_to_array
classes=['B1','B2','B3','B4','B5','B7']                                                                       #更改處
model = load_model('ClassifylilData.h5')                                                         #更改處
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

files = os.listdir("lilTest")
accuracy = 0
totalcount = 0
for f in files:
 img = image.load_img("lilTest\\"+f, target_size=(128, 128))
 totalcount+=1
 x = img_to_array(img)
 x = np.expand_dims(x, axis=0)
 images = np.vstack([x])
 result = model.predict(images)
 ind=np.argmax(result,1)

 filebasename = f.split("_")[0]
 if(filebasename == classes[ind[0]]):
  accuracy+=1
 print('this '+f+' is a ', classes[ind[0]])
print("Test accuracy : ",round((accuracy/totalcount),2))
