import nibabel as nib
import numpy as np
import os

nifti_path = '../datasets/gad/nifti/'
npy_path = '../datasets/gad/npy/'

for x in ('pre', 'post'):
    for filename in os.listdir(os.path.join(nifti_path, x)):
        path = os.path.join(nifti_path, x, filename)
        subj_name, ext = os.path.splitext(filename)
        if ext == '.gz':
            subj_name = os.path.splitext(subj_name)[0]
        
        #Load image
        img = nib.load(path)
        affine = img.affine
        data = img.get_data()
        
        #Transpose so that acquisition slices are in the first dimension
        data = data.transpose(2, 0, 1)
        
        #Add channel dimension
        data = np.expand_dims(data, axis=3)
        
        #Save affine, image as .npy
        print('Saving {:s}-{:s}...'.format(x, subj_name))
        np.save(os.path.join(npy_path, 'affine', x, subj_name), affine)
        np.save(os.path.join(npy_path, 'whole', x, subj_name), data)
        
        #Save slices as .npy
        for i in range(data.shape[-1]):
            sl_name = '{:s}-{:03d}'.format(subj_name, i)
            np.save(os.path.join(npy_path, 'slices', x, sl_name), 
                    data[:, :, i, :])

print('Done.')

