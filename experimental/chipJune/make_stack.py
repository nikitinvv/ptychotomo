import tomopy
import dxchange
import numpy as np
import h5py
import sys
import skimage.feature
from tomoalign import solver_deform
from tomoalign import utils
##################################### Inputs #########################################################################
sino_start = 0
sino_end = 2048
flat_field_norm = True
flat_field_drift_corr = False  # Correct the intensity drift
remove_rings = False
binning=0
######################################################################################################################


def preprocess_data(prj, flat, dark, FF_norm=flat_field_norm, remove_rings=remove_rings, FF_drift_corr=flat_field_drift_corr, downsapling=binning):

    if FF_norm:  # dark-flat field correction
        prj = tomopy.normalize(prj, flat, dark)
    if FF_drift_corr:  # flat field drift correction
        prj = tomopy.normalize_bg(prj, air=50)
    prj[prj <= 0] = 1  # check dark<data
    prj = tomopy.minus_log(prj)  # -logarithm
    if remove_rings:  # remove rings
        prj = tomopy.remove_stripe_fw(
             prj, level=7, wname='sym16', sigma=1, pad=True)
        #prj = tomopy.remove_stripe_ti(prj,2)
        # prj = tomopy.remove_all_stripe(prj)
    if downsapling > 0:  # binning
        prj = tomopy.downsample(prj, level=binning)
        prj = tomopy.downsample(prj, level=binning, axis=1)
    return prj


if __name__ == "__main__":
    name = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/chipJune/chip_16nmZP_tube_lens_interlaced_2000prj_3s_'
    strs = ['098','099','100','101','102','103','104','105']#,'106','107']
    ang = int(sys.argv[1])
    arr = np.zeros([len(strs),2048,2448],dtype='float32')
    for j in range(len(strs)):                
        name0 = name+strs[j]+'.h5'
        print(name0)
    # read data
        prj, flat, dark, theta = dxchange.read_aps_32id(
            name0, sino=(sino_start, sino_end), proj=(ang,ang+1))
        # preprocess
        prj = preprocess_data(prj, flat, dark, FF_norm=flat_field_norm, remove_rings=remove_rings,
                            FF_drift_corr=flat_field_drift_corr, downsapling=binning)

        dxchange.write_tiff(prj[0],name+'/theta'+str(ang)+'/'+str(j),overwrite=True)
        arr[j]=prj[0]

    mmin, mmax = utils.find_min_max(prj[:1])
    mmin[:]=0
    mmin[:]=0.1
    mmax[:]=2
    # parameters for non-dense flow in Farneback's algorithm,
    # resulting flow is constant, i.e. equivalent to a shift
    res = arr.copy()
    pars = [0.5, 4, 12, 16, 5, 1.1, 0]
    print(mmin,mmax)
    for k in range(arr.shape[0]):
        with solver_deform.SolverDeform(1, 2048, 2448, 1, 1) as dslv:
            print(np.linalg.norm(arr[k:k+1]))
            print(np.linalg.norm(arr[0:1]))
            t1 = arr[k:k+1].copy()
            t2 = arr[0:1].copy() 
            flow = dslv.registration_flow_batch(
                t1, t2, mmin, mmax, None, pars)
            print(np.linalg.norm(flow))
            res[k:k+1] = dslv.apply_flow_gpu_batch(arr[k:k+1], flow)
    dxchange.write_tiff_stack(res,name+'/theta'+str(ang)+'aligned/r',overwrite=True)            
    