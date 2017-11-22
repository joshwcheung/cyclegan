import nibabel as nib
import numpy as np
import os

from glob import glob

def npy_to_nifti(subject, npy_path, affine_path, nifti_path, basename):
    pattern = '{:s}*.npy'.format(basename)
    image_paths = sorted(glob(os.path.join(npy_path, pattern)))
    affine_path = os.path.join(affine_path, '{:s}.npy'.format(subject))
    
    affine = np.load(affine_path)
    image_list = []
    for path in image_paths:
        image = np.squeeze(np.load(path))
        image_list.append(image)
    
    array_data = np.stack(image_list, axis=-1)
    array_img = nib.Nifti1Image(array_data, affine)
    out_path = os.path.join(nifti_path, '{:s}.nii.gz'.format(basename))
    nib.save(array_img, out_path)

def main():

    nifti_path = '../datasets/gad/nifti/'
    npy_path = '../datasets/gad/npy/'

    for x in ('pre', 'post'):
        for filename in os.listdir(os.path.join(nifti_path, x)):
            path = os.path.join(nifti_path, x, filename)
            subj_name, ext = os.path.splitext(filename)
            if ext == '.gz':
                subj_name = os.path.splitext(subj_name)[0]
            
            affine_path = os.path.join(npy_path, 'affine', x)
            whole_path = os.path.join(npy_path, 'whole', x)
            slices_path = os.path.join(npy_path, 'slices', x)
            if not os.path.exists(affine_path):
                os.makedirs(affine_path)
            if not os.path.exists(whole_path):
                os.makedirs(whole_path)
            if not os.path.exists(slices_path):
                os.makedirs(slices_path)
            
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
            np.save(os.path.join(affine_path, subj_name), affine)
            np.save(os.path.join(whole_path, subj_name), data)
            
            #Save slices as .npy
            for i in range(data.shape[0]):
                slice_name = '{:s}-{:03d}'.format(subj_name, i)
                np.save(os.path.join(slices_path, slice_name), 
                        data[i, :, :, :])
    print('Done.')

if __name__ == '__main__':
    main()
