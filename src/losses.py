import tensorflow as tf


# defining gram martix to calculate style loss
def gram_matrix(x):
    features = tf.reshape(tf.transpose(x, (2, 0, 1)), (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


def content_loss(content, combination):
    """
    The content loss is used to measure the content consistency 
    between the output image and the content image.
    """
    return tf.reduce_sum(tf.square(combination - content))


def style_loss(style, combination, size):
    """
    The style loss is used to measure the style consistency
    between the output image and the style image.
    """
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


def total_variation_loss(combination_image, image_height, image_width):
    """
    Total variation loss can promote the model to produce
    a smooth image
    """
    a = tf.square(combination_image[:, :image_height-1, :image_width -
                                    1, :] - combination_image[:, 1:, :image_width-1, :])
    b = tf.square(combination_image[:, :image_height-1, :image_width -
                                    1, :] - combination_image[:, :image_height-1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))
