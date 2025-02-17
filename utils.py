import numpy as np
import sys
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()
def calc_mean_IOU(pred, gt, thresh):
    # slightly modified from https://github.com/chrischoy/3D-R2N2/blob/master/lib/voxel.py
    pred_s = np.squeeze(pred[:, 1, :, :, :])
    preds_occupy = pred_s >= thresh
    diff = np.sum(np.logical_xor(preds_occupy, gt))
    intersection = np.sum(np.logical_and(preds_occupy, gt), (1,2,3))
    union = np.sum(np.logical_or(preds_occupy, gt), (1,2,3))
    num_fp = np.sum(np.logical_and(preds_occupy, np.logical_not(gt)))  # false positive
    num_fn = np.sum(np.logical_and(np.logical_not(preds_occupy), gt))  # false negative
    iou = np.mean(intersection.astype('float32')/union.astype('float32'))
    return np.array([diff, np.sum(intersection), np.sum(union), num_fp, num_fn, iou])

def voxel2mesh(voxels, surface_view):
    # taken from https://github.com/chrischoy/3D-R2N2/blob/master/lib/voxel.py
    cube_verts = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0],
                  [1, 1, 1]]  # 8 points

    cube_faces = [[0, 1, 2], [1, 3, 2], [2, 3, 6], [3, 7, 6], [0, 2, 6], [0, 6, 4], [0, 5, 1],
                  [0, 4, 5], [6, 7, 5], [6, 5, 4], [1, 7, 3], [1, 5, 7]]  # 12 face

    cube_verts = np.array(cube_verts)
    cube_faces = np.array(cube_faces) + 1

    scale = 0.01
    cube_dist_scale = 1.1
    verts = []
    faces = []
    curr_vert = 0

    positions = np.where(voxels > 0.3)
    voxels[positions] = 1 
    for i, j, k in zip(*positions):
        # identifies if current voxel has an exposed face 
        if not surface_view or np.sum(voxels[i-1:i+2, j-1:j+2, k-1:k+2]) < 27:
            verts.extend(scale * (cube_verts + cube_dist_scale * np.array([[i, j, k]])))
            faces.extend(cube_faces + curr_vert)
            curr_vert += len(cube_verts)  
              
    return np.array(verts), np.array(faces)
    
    
def write_obj(filename, verts, faces):
    # taken from https://github.com/chrischoy/3D-R2N2/blob/master/lib/voxel.py:
    """ write the verts and faces on file."""
    with open(filename, 'w') as f:
        # write vertices
        f.write('g\n# %d vertex\n' % len(verts))
        for vert in verts:
            f.write('v %f %f %f\n' % tuple(vert))

        # write faces
        f.write('# %d faces\n' % len(faces))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face))

    
def voxel2obj(filename, pred, surface_view = True):
    # taken from https://github.com/chrischoy/3D-R2N2/blob/master/lib/voxel.py
    verts, faces = voxel2mesh(pred, surface_view)
    write_obj(filename, verts, faces)
