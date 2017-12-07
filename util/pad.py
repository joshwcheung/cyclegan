import tensorflow as tf

def _is_tensor(x):
    return isinstance(x, (tf.Tensor, tf.Variable))

def _ImageDimensions(image, rank):
    if image.get_shape().is_fully_defined():
        return image.get_shape().as_list()
    else:
        static_shape = image.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(image), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]

def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width, mode='REFLECT'):
    #Modified from TensorFlow function tf.image.pad_to_bounding_box
    image = tf.convert_to_tensor(image, name='image')
    is_batch = True
    image_shape = image.get_shape()
    if image_shape.ndims == 3:
        is_batch = False
        image = tf.expand_dims(image, 0)
    elif image_shape.ndims is None:
        is_batch = False
        image = tf.expand_dims(image, 0)
        image.set_shape([None] * 4)
    elif image_shape.ndims != 4:
        raise ValueError('\'image\' must have either 3 or 4 dimensions.')
    
    batch, height, width, depth = _ImageDimensions(image, rank=4)
    after_padding_width = target_width - offset_width - width
    after_padding_height = target_height - offset_height - height
    
    paddings = tf.reshape(tf.stack([0, 0, offset_height, after_padding_height, 
                                    offset_width, after_padding_width, 0, 0]), 
                          [4, 2])
    padded = tf.pad(image, paddings, mode=mode)
    
    padded_shape = [None if _is_tensor(i) else i 
                    for i in [batch, target_height, target_width, depth]]
    padded.set_shape(padded_shape)
    
    if not is_batch:
        padded = tf.squeeze(padded, squeeze_dims=[0])
    
    return padded

def resize_pad(image, target_height, target_width, mode='REFLECT'):
    #Modified from TensorFLow function tf.image.resize_image_with_crop_or_pad
    image = tf.convert_to_tensor(image, name='image')
    image_shape = image.get_shape()
    is_batch = True
    if image_shape.ndims == 3:
        is_batch = False
        image = tf.expand_dims(image, 0)
    elif image_shape.ndims is None:
        is_batch = False
        image = tf.expand_dims(image, 0)
        image.set_shape([None] * 4)
    elif image_shape.ndims != 4:
        raise ValueError('\'image\' must have either 3 or 4 dimensions.')
    
    def max_(x, y):
        if _is_tensor(x) or _is_tensor(y):
            return tf.maximum(x, y)
        else:
            return max(x, y)
    
    _, height, width, _ = _ImageDimensions(image, rank=4)
    width_diff = target_width - width
    offset_crop_width = max_(-width_diff // 2, 0)
    offset_pad_width = max_(width_diff // 2, 0)
    
    height_diff = target_height - height
    offset_crop_height = max_(-height_diff // 2, 0)
    offset_pad_height = max_(height_diff // 2, 0)
    
    resized = pad_to_bounding_box(image, offset_pad_height, offset_pad_width, 
                                  target_height, target_width, mode=mode)
    
    if not is_batch:
        resized = tf.squeeze(resized, squeeze_dims=[0])
    
    return resized

