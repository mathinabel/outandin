import os
import glob
import argparse

from PIL import Image

# Keras / TensorFlow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from indoornet.model.layers import BilinearUpSampling2D
from indoornet.dataandgeo.utils.utils import predict, load_images, display_images
from matplotlib import pyplot as plt


def input_transform():
    im = Image.open("examples/95.jpg" )
    im = im.resize((640, 480), Image.BILINEAR)  # 第一个参数为想要的size，第二个参数为插值方法，双线性插值这里用的是
    im.save('{}/{}.png'.format("examples", "p13"))

input_transform()
# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/95.jpg', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

# Input images
inputs = load_images(glob.glob(args.input))
viz = display_images(inputs, inputs)
plt.figure(figsize=(10, 5))
plt.imshow(viz)
print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results
outputs = predict(model, inputs)

# matplotlib problem on ubuntu terminal fix
# matplotlib.use('TkAgg')

# Display results
viz = display_images(outputs.copy(), inputs.copy())
plt.figure(figsize=(10, 5))
plt.imshow(viz)
plt.savefig('test.png')
plt.show()
