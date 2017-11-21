import nibabel as nib
import numpy as np
import os
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def np_to_binary(array):
    shape = np.array(array.shape, np.int32)
    return shape.tobytes(), array.tobytes()

def write_to_tfrecord(array, tfrecord_file):
    shape, binary = np_to_binary(array)
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    features = tf.train.Features(feature={'shape': _bytes_feature(shape), 
                                          'array': _bytes_feature(binary)})
    example = tf.train.Example(features=features)
    writer.write(example.SerializeToString())
    writer.close()

def read_from_tfrecord(filenames, shuffle=True):
    queue = tf.train.string_input_producer(filenames, shuffle=shuffle, 
                                           name='queue')
    reader = tf.TFRecordReader()
    _, record = reader.read(queue)
    
    features = {'shape': tf.FixedLenFeature([], tf.string), 
                'array': tf.FixedLenFeature([], tf.string)}
    tfrecord_features = tf.parse_single_example(record, features=features, 
                                                name='features')
    array = tf.decode_raw(tfrecord_features['array'], tf.float32)
    shape = tf.decode_raw(tfrecord_features['shape'], tf.int32)
    array = tf.reshape(array, shape)
    return array

def main():
    nifti_path = '../datasets/gad/nifti/'
    tfrecord_path = '../datasets/gad/tfrecord/'
    for x in ('pre', 'post'):
        for filename in os.listdir(os.path.join(nifti_path, x)):
            path = os.path.join(nifti_path, x, filename)
            subj_name, ext = os.path.splitext(filename)
            if ext == '.gz':
                subj_name = os.path.splitext(subj_name)[0]
            
            affine_path = os.path.join(tfrecord_path, 'affine', x)
            whole_path = os.path.join(tfrecord_path, 'whole', x)
            slices_path = os.path.join(tfrecord_path, 'slices', x)
            if not os.path.exists(affine_path):
                os.makedirs(affine_path)
            if not os.path.exists(whole_path):
                os.makedirs(whole_path)
            if not os.path.exists(slices_path):
                os.makedirs(slices_path)
            
            #Load image
            img = nib.load(path)
            affine = np.asarray(img.affine, dtype=np.float32)
            data = img.get_data()
            
            #Transpose so that acquisition slices are in the first dimension
            data = data.transpose(2, 0, 1)
            
            #Add channel dimension
            data = np.expand_dims(data, axis=3)
            
            #Save affine, image as .tfrecord
            print('Saving {:s}-{:s}...'.format(x, subj_name))
            out = os.path.join(affine_path, '{:s}.tfrecord'.format(subj_name))
            write_to_tfrecord(affine, out)
            out = os.path.join(whole_path, '{:s}.tfrecord'.format(subj_name))
            write_to_tfrecord(data, out)
            
            #Save slices as .tfrecord
            for i in range(data.shape[0]):
                slice_name = '{:s}-{:03d}'.format(subj_name, i)
                out = os.path.join(slices_path, 
                                   '{:s}.tfrecord'.format(slice_name))
                write_to_tfrecord(data[i, :, :, :], out)
    print('Done.')

if __name__ == '__main__':
    main()

