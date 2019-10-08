# Listing 5.4 Copying images to training, validation, and test directories

import os, shutil
base_dir = '@dataset/catdog/'


# In[2]:


train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')


# In[3]:


print('total training cat images:', len(os.listdir(train_cats_dir)))


# In[4]:


print('total training dog images:', len(os.listdir(train_dogs_dir)))


# In[5]:


print('total validation cat images:', len(os.listdir(validation_cats_dir)))


# In[6]:


print('total validation dog images:', len(os.listdir(validation_dogs_dir)))


# In[7]:


print('total test cat images:', len(os.listdir(test_cats_dir)))


# In[8]:


print('total test dog images:', len(os.listdir(test_dogs_dir)))


# In[9]:


# Listing 5.5 Instantiating a small convnet for dogs vs. cats classification

from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[10]:


model.summary()


# In[11]:


# Listing 5.6 Configuring the model for training

from keras import optimizers
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


# In[12]:


# Listing 5.7 Using ImageDataGenerator to read images from directories

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


# In[13]:


train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='binary')


# In[14]:


validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')


# In[15]:


# Listing 5.8 Fitting the model using a batch generator

history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=50)


# In[16]:


# Listing 5.9 Saving the model
model.save('catdog1.h5')


# In[18]:


# Listing 5.10 Displaying curves of loss and accuracy during training

import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[19]:


# Listing 5.11 Setting up a data augmentation configuration via ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')


# In[22]:


# Listing 5.12 Displaying some randomly augmented training images

from keras.preprocessing import image

fnames = [os.path.join(train_cats_dir, fname) for
        fname in os.listdir(train_cats_dir)]

img_path = fnames[5]

img = image.load_img(img_path, target_size=(150, 150))

x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()


# In[23]:


# Listing 5.13 Defining a new convnet that includes dropout

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
            optimizer=optimizers.RMSprop(lr=1e-4),
            metrics=['acc'])


# In[24]:


# Listing 5.14 Training the convnet using data-augmentation generators

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=50)


# In[25]:


# Listing 5.15 Saving the model

model.save('catdog2.h5')


# In[26]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[27]:


# Listing 5.16 Instantiating the VGG16 convolutional base

from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                    include_top=False,
                    input_shape=(150, 150, 3))


# In[28]:


conv_base.summary()


# In[29]:


# Listing 5.17 Extracting features using the pretrained convolutional base

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# base_dir = '/Users/fchollet/Downloads/cats_and_dogs_small'
base_dir = 'd:/@dataset/catdog'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(directory,
                                            target_size=(150, 150),
                                            batch_size=batch_size,
                                            class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

# The extracted features are currently of shape (samples, 4, 4, 512). You’ll feed them
# to a densely connected classifier, so first you must flatten them to (samples, 8192):

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))


# In[30]:


# Listing 5.18 Defining and training the densely connected classifier

from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                loss='binary_crossentropy',
                metrics=['acc'])

model.summary()

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))


# In[31]:


# Listing 5.19 Plotting the results

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# # FEATURE EXTRACTION WITH DATA AUGMENTATION
# Now, let’s review the second technique I mentioned for doing feature extraction,
# which is much slower and more expensive, but which allows you to use data augmentation
# during training: extending the conv_base model and running it end to end on
# the inputs.

# In[ ]:


from keras import models
from keras import layers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()


# In[33]:


print('This is the number of trainable weights '
      'before freezing the conv base:', len(model.trainable_weights))


# In[35]:


conv_base.trainable = False
print('This is the number of trainable weights '
      'after freezing the conv base:', len(model.trainable_weights))


# In[36]:


# Listing 5.21 Training the model end to end with a frozen convolutional base

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255) # Note that the validation data shouldn’t be augmented!

train_generator = train_datagen.flow_from_directory(train_dir,  # Traget directory 
                                    target_size=(150, 150),   # Resize all images to 150 x 150
                                    batch_size=20,
                                    class_mode='binary') # Because you use binary_crossentropy loss, you need binary lables

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                    target_size=(150, 150),
                                    batch_size=20,
                                    class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=50)


# In[37]:


# Let’s plot the results again (see figures 5.17 and 5.18). As you can see, you reach a validation accuracy of about 96%.

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# # Fine-tuning
# Another widely used technique for model reuse, complementary to feature
# extraction, is fine-tuning (see figure 5.19). Fine-tuning consists of unfreezing a few of
# the top layers of a frozen model base used for feature extraction, and jointly training
# both the newly added part of the model (in this case, the fully connected classifier)
# and these top layers. This is called fine-tuning because it slightly adjusts the more
# abstract representations of the model being reused, in order to make them more relevant
# for the problem at hand.

# In[38]:


conv_base.summary()


# In[39]:


# Listing 5.22 Freezing all layers up to a specific one

conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


# Now you can begin fine-tuning the network. You’ll do this with the RMSProp optimizer,
# using a very low learning rate. The reason for using a low learning rate is that
# you want to limit the magnitude of the modifications you make to the representations
# of the three layers you’re fine-tuning. Updates that are too large may harm these representations.

# In[40]:


# Listing 5.23 Fine-tuning the model

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

history = model.fit_generator(train_generator,
                             steps_per_epoch=100,
                             epochs=100,
                             validation_data=validation_generator,
                             validation_steps=50)


# In[41]:


# Let’s plot the results using the same plotting code as before (see figures 5.20 and 5.21).

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# These curves look noisy. To make them more readable, you can smooth them by
# replacing every loss and accuracy with exponential moving averages of these quantities.
# Here’s a trivial utility function to do this (see figures 5.22 and 5.23).

# In[42]:


# Listing 5.24 Smoothing the plots

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

plt.plot(epochs,
        smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs,
        smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,
        smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs,
        smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[43]:


test_generator = test_datagen.flow_from_directory(test_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)


# # 5.3.3 Wrapping up
# Here’s what you should take away from the exercises in the past two sections:
# 
#  * Convnets are the best type of machine-learning models for computer-vision tasks. It’s possible to train one from scratch even on a very small dataset, with decent results.
#  * On a small dataset, overfitting will be the main issue. Data augmentation is a powerful way to fight overfitting when you’re working with image data.
#  * It’s easy to reuse an existing convnet on a new dataset via feature extraction. This is a valuable technique for working with small image datasets.
#  * As a complement to feature extraction, you can use fine-tuning, which adapts to a new problem some of the representations previously learned by an existing model. This pushes performance a bit further. Now you have a solid set of tools for dealing with image-classification problems—in particular with small datasets.

# # 5.4 Visualizing what convnets learn
# * Visualizing intermediate convnet outputs (intermediate activations)
# * Visualizing convnets filters
# * Visualizing heatmaps of class activation in an image
# 
# ## 5.4.1 Visualizing intermediate activations

# In[48]:


from keras.models import load_model
model = load_model('catdog2.h5')
model.summary() 


# In[49]:


# Listing 5.25 Preprocessing a single image

img_path = 'd:/@dataset/catdog/test/cats/cat.4508.jpg'

from keras.preprocessing import image # Preprocesses the image into a 4D tensor
import numpy as np

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.  # Remember that the model was trained on inputs that were preprocessed this way.

print(img_tensor.shape)


# In[50]:


# Listing 5.26 Displaying the test picture

import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.show()


# In[51]:


# Listing 5.27 Instantiating a model from an input tensor and a list of output tensors

from keras import models

layer_outputs = [layer.output for layer in model.layers[:8]] # Extracts the outputs of the top eight layers
activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, 
# given the model input


# In[52]:


# Listing 5.28 Running the model in predict mode

activations = activation_model.predict(img_tensor) # Returns a list of five Numpy arrays: one array per layer activation


# In[53]:


first_layer_activation = activations[0]
print(first_layer_activation.shape)


# In[68]:


# Listing 5.29 Visualizing the fourth channel

import matplotlib.pyplot as plt
plt.matshow(first_layer_activation[0, :, :, 11], cmap='viridis')


# In[74]:


# Listing 5.31 Visualizing every channel in every intermediate activation

layer_names = []
for layer in model.layers[:7]:     # Names of the layers, so you can have them as part of your plot
    layer_names.append(layer.name)
    
images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    
    size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
    
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                            :, :,
                                            col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                        row * size : (row + 1) * size] = channel_image
            
scale = 1. / size
plt.figure(figsize=(scale * display_grid.shape[1],
                    scale * display_grid.shape[0]))
plt.title(layer_name)
plt.grid(False)
plt.imshow(display_grid, aspect='auto', cmap='viridis')


# # 5.4.2 Visualizing convnet filters

# In[75]:


# Listing 5.32 Defining the loss tensor for filter visualization

from keras.applications import VGG16
from keras import backend as K

model = VGG16(weights='imagenet',
                include_top=False)

layer_name = 'block3_conv1'
filter_index = 0

layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])


# In[76]:


# Listing 5.33 Obtaining the gradient of the loss with regard to the input

grads = K.gradients(loss, model.input)[0]


# In[77]:


# Listing 5.34 Gradient-normalization trick

grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)


# In[78]:


# Listing 5.35 Fetching Numpy output values given Numpy input values

iterate = K.function([model.input], [loss, grads])

import numpy as np
loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])


# In[82]:


# Listing 5.36 Loss maximization via stochastic gradient descent

input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.  # Starts from a gray image with some noise

step = 1.               # Magnitude of each gradient update
for i in range(40):
            loss_value, grads_value = iterate([input_img_data])
        
            input_img_data += grads_value * step


# In[83]:


# Listing 5.37 Utility function to convert a tensor into a valid image

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    
    x += 0.5
    x = np.clip(x, 0, 1)
    
    x *= 255    # Converts to an RGB array
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# In[84]:


# Listing 5.38 Function to generate filter visualizations

def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img = input_img_data[0]
    return deprocess_image(img)


# In[85]:


plt.imshow(generate_pattern('block3_conv1', 0))


# In[88]:


# Listing 5.39 Generating a grid of all filter response patterns in a layer

layer_name = 'block2_conv1'
size = 64
margin = 5

results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

for i in range(8):
    for j in range(8):
        filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
        
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end,
                            vertical_start: vertical_end, :] = filter_img
        
plt.figure(figsize=(20, 20))
plt.imshow(results)


# # 5.4.3 Visualizing heatmaps of class activation

# In[89]:


# Listing 5.40 Loading the VGG16 network with pretrained weights

from keras.applications.vgg16 import VGG16
model = VGG16(weights='imagenet')


# In[90]:


# Listing 5.41 Preprocessing an input image for VGG16

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
# img_path = '/Users/fchollet/Downloads/creative_commons_elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


# In[91]:


preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])


# In[92]:


np.argmax(preds[0])


# In[93]:


# Listing 5.42 Setting up the Grad-CAM algorithm

siamese_cat_output = model.output[:, 284]

last_conv_layer = model.get_layer('block5_conv3')

grads = K.gradients(siamese_cat_output, last_conv_layer.output)[0]

pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input],
                    [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    
heatmap = np.mean(conv_layer_output_value, axis=-1)


# In[94]:


# Listing 5.43 Heatmap post-processing

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)


# In[95]:


# Listing 5.44 Superimposing the heatmap with the original picture

import cv2

img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('d:/@dataset/siamese.jpg', superimposed_img)


# ## Chapter summary
# * Convnets are the best tool for attacking visual-classification problems.
# * Convnets work by learning a hierarchy of modular patterns and concepts to represent the visual world.
# * The representations they learn are easy to inspect—convnets are the opposite of black boxes!
# * You’re now capable of training your own convnet from scratch to solve an image-classification problem.
# * You understand how to use visual data augmentation to fight overfitting.
# * You know how to use a pretrained convnet to do feature extraction and fine-tuning.
# * You can generate visualizations of the filters learned by your convnets, as well as heatmaps of class activity.
