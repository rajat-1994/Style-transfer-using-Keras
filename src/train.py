import cv2
from utils import denormalize_image, preprocess_image
from losses import content_loss, total_variation_loss, style_loss
from tensorflow import keras
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.applications import vgg19

CONTENT_WEIGHT = 1e-9
STYLE_WEIGHT = 1e-6
TV_WEIGHT = 4e-6

base_image_path = './images/hugo.jpg'
style_image_path = './images/picasso.jpg'
file_prefix = "./outputs/hugo_picasso"

base_img = cv2.imread(base_image_path)
print("shape of base image", base_img.shape)

# Defining output image dimensions
image_height = 512
image_width = int(base_img.shape[1] * image_height / base_img.shape[0])

size = (image_width, image_height)

model = vgg19.VGG19(weights='imagenet', include_top=False)
print("Vgg19 loaded")

feature_layers = dict([(layer.name, layer.output)
                       for layer in model.layers])

feature_model = keras.Model(inputs=model.inputs, outputs=feature_layers)

style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]
# The layer to use for the content loss.
content_layer_name = "block2_conv2"


def total_loss(combination_image, base_image, style_image):
    input_tensor = tf.concat(
        [base_image, style_image, combination_image], axis=0
    )
    features = feature_model(input_tensor)

    # Initialize the loss
    loss = tf.zeros(shape=())

    layer_features = features[content_layer_name]
    content_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]

    loss = loss + CONTENT_WEIGHT * content_loss(content_image_features,
                                                combination_features)

    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_features, combination_features,
                        size=image_height*image_width)
        loss += (STYLE_WEIGHT / len(style_layer_names)) * sl

    loss += TV_WEIGHT * \
        total_variation_loss(combination_image, image_height, image_width)
    return loss


@tf.function
def compute_loss_and_grads(combination_image,
                           base_image,
                           style_reference_image):
    with tf.GradientTape() as tape:
        loss = total_loss(combination_image, base_image, style_reference_image)
    grads = tape.gradient(loss, combination_image)
    return loss, grads


optimizer = keras.optimizers.SGD(
    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=100.0, decay_steps=100, decay_rate=0.99
    )
)

base_image = preprocess_image(base_image_path, size)
style_reference_image = preprocess_image(style_image_path, size)
combination_image = tf.Variable(preprocess_image(base_image_path, size))
print(base_image.shape, style_reference_image.shape, combination_image.shape)
print("Staring Traning")
iterations = 4000
for i in tqdm(range(1, iterations + 1)):
    loss, grads = compute_loss_and_grads(
        combination_image, base_image, style_reference_image
    )
    optimizer.apply_gradients([(grads, combination_image)])
    if i % 500 == 0:
        print("Iteration %d: loss=%.2f" % (i, loss))
        img = denormalize_image(combination_image.numpy(),
                                (image_width, image_height, 3))
        fname = file_prefix + "_at_iteration_%d.png" % i
        cv2.imwrite(fname, img)
