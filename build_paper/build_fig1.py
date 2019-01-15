import objects
import solver_gpu
import dxchange
import tomopy
import numpy as np
import signal
import sys
import os
import glob
import scipy.misc


if __name__ == "__main__":

    # Parameters.
    rho = 0.5
    gamma = 0.25
    eta = 0.25
    piter = 4
    titer = 4
    NITER = 50

    piters = 200
    titers = 200
    NITERs = 1

    voxelsize = 1e-6
    energy = 5
    prbsize = 15

    # Load a 3D object
    beta = dxchange.read_tiff(
        'data/test-beta-128.tiff').astype('float32')[0:30:2, ::2, ::2]
    delta = dxchange.read_tiff(
        'data/test-delta-128.tiff').astype('float32')[0:30:2, ::2, ::2]

    # Create object.
    obj = objects.Object(beta, delta, voxelsize)
    # Detector parameters
    det = objects.Detector(63, 63)
    # Define rotation angles
    theta = np.linspace(0, 2*np.pi, 200).astype('float32')
    # tomography data shape
    tomoshape = [len(theta), obj.shape[0], obj.shape[2]]

    maxinta = [3, 3, 1, 0.3, 0.1]
    shift = 4
    sigmaa = np.zeros(5, dtype=np.int)

    for k in range(5):
        # Create probe
        prb = objects.Probe(objects.gaussian(
            prbsize, rin=0.8, rout=1.0), maxint=maxinta[k])
        # Scanner positions
        scanax, scanay = objects.scanner3(theta, beta.shape, shift, shift, margin=[
            prb.size, prb.size], offset=[0, 0], spiral=1)
        # Class solver
        slv = solver_gpu.Solver(prb, scanax, scanay,
                                theta, det, voxelsize, energy, tomoshape)

        def signal_handler(sig, frame):
            slv = []
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTSTP, signal_handler)
        # Compute data  |FQ(exp(i\nu R x))|^2,
        data = np.abs(slv.fwd_ptycho(
            slv.exptomo(slv.fwd_tomo(obj.complexform))))**2
        print("sigma = ", np.amax(np.sqrt(data*det.x*det.y)))
        sigmaa[k] = np.amax(np.sqrt(data*det.x*det.y))

        # Apply Poisson noise (warning: Poisson distribution is discrete, so the resulting values are integers)
        if(k > 0):
            data = np.random.poisson(
                data*det.x*det.y).astype('float32')/(det.x*det.y)
        
#############joint (rho=0.5)
        rho = 0.5
        # Initial guess        
        h = np.ones(tomoshape, dtype='complex64', order='C')
        psi = np.ones(tomoshape, dtype='complex64', order='C')
        lamd = np.zeros(tomoshape, dtype='complex64', order='C')
        x = objects.Object(np.zeros(obj.shape, dtype='float32', order='C'), np.zeros(
            obj.shape, dtype='float32', order='C'), voxelsize)

        # ADMM
        x = slv.admm(data, h, psi, lamd, x, rho,
                     gamma, eta, piter, titer, NITER)

        # Save result
        name = '_noise'+np.str(k > 0)+'_ang'+str(len(theta))+'_shift'+str(shift) + \
            '_maxint'+str(maxinta[k])
        dxchange.write_tiff(
            x.beta[8],  'fig1/tiff_png/beta_joint'+name, overwrite=True)
        dxchange.write_tiff(
            x.delta[8],  'fig1/tiff_png/delta_joint'+name, overwrite=True)

# #############joint (rho=0)
        rho = 1e-13
        # Initial guess
        h = np.ones(tomoshape, dtype='complex64', order='C')
        psi = np.ones(tomoshape, dtype='complex64', order='C')
        lamd = np.zeros(tomoshape, dtype='complex64', order='C')
        x = objects.Object(np.zeros(obj.shape, dtype='float32', order='C'), np.zeros(
            obj.shape, dtype='float32', order='C'), voxelsize)

        # ADMM
        x = slv.admm(data, h, psi, lamd, x, rho,
                     gamma, eta, piter, titer, NITER)
        slv=[]

        # Save result
        name = '_noise'+np.str(k > 0)+'_ang'+str(len(theta))+'_shift'+str(shift) + \
            '_maxint'+str(maxinta[k])
        dxchange.write_tiff(
            x.beta[8],  'fig1/tiff_png/beta_joint0'+name, overwrite=True)
        dxchange.write_tiff(
            x.delta[8],  'fig1/tiff_png/delta_joint0'+name, overwrite=True)


# #############seq
        rho = 1e-13
        slv = solver_gpu.Solver(prb, scanax, scanay,
                                theta, det, voxelsize, energy, tomoshape)
        # Initial guess
        h = np.ones(tomoshape, dtype='complex64', order='C')
        psi = np.ones(tomoshape, dtype='complex64', order='C')
        lamd = np.zeros(tomoshape, dtype='complex64', order='C')
        x = objects.Object(np.zeros(obj.shape, dtype='float32', order='C'), np.zeros(
            obj.shape, dtype='float32', order='C'), voxelsize)

        # ADMM
        x = slv.admm(data, h, psi, lamd, x, rho,
                     gamma, eta, piters, titers, NITERs)

        # Save result
        name = '_noise'+np.str(k > 0)+'_ang'+str(len(theta))+'_shift'+str(shift) + \
            '_maxint'+str(maxinta[k])
        dxchange.write_tiff(
            x.beta[8],  'fig1/tiff_png/beta_st'+name, overwrite=True)
        dxchange.write_tiff(
            x.delta[8],  'fig1/tiff_png/delta_st'+name, overwrite=True)

    files = glob.glob('fig1/tiff_png/delta*.tiff')

    for name in files:
        a = dxchange.read_tiff(name)
        scipy.misc.toimage(a, cmin=-0.00013467284/4, cmax=0.00013467284).save(
            str(os.path.splitext(name)[0])+'.png')
    files = glob.glob('fig1/tiff_png/beta*.tiff')
    for name in files:
        a = dxchange.read_tiff(name)
        scipy.misc.toimage(a, cmin=-2.7094511e-05/4, cmax=2.7094511e-05).save(
            str(os.path.splitext(name)[0])+'.png')
    os.system(
        'for f in fig1/tiff_png/*.png; do  convert -trim "$f" "$f"; done')

    fid = open('fig1/fig1.tex', 'w')
    fid.write("\\begin{figure*}\n \\begin{tabular}{MM*5{C}@{}}\n & & noise free  & $\\sigma_1=%d$ & $\\sigma_2=%d$ & $\\sigma_3=%d$ & $\\sigma_4=%d$ \\\\ \n" %
              (sigmaa[1], sigmaa[2], sigmaa[3], sigmaa[4]))

    # beta
    fid.write(
        "\\multirow{2}{*}{\\rotatebox[origin=c]{90}{beta reconstruction\\hspace*{1cm} }}")
    # standard
    fid.write(" & \\rotatebox[origin=c]{90}{ standard } ")
    for k in range(0, 5):            
        name = ' & \\includegraphics[width=0.175\\textwidth]{{fig1/beta_st_noise'+np.str(k > 0)+'_ang'+str(len(theta))+'_shift'+str(shift) + \
            '_maxint'+str(maxinta[k])+'}.png}'
        fid.write(name)
    fid.write("\\\\ \n")
    # joint 0
    fid.write("& \\rotatebox[origin=c]{90}{ coupled } ")
    for k in range(0, 5):
        name = ' & \\includegraphics[width=0.175\\textwidth]{{fig1/beta_joint0_noise'+np.str(k > 0)+'_ang'+str(len(theta))+'_shift'+str(shift) + \
            '_maxint'+str(maxinta[k])+'}.png}'
        fid.write(name)
    fid.write("\\\\ \n")
    # joint
    fid.write("& \\rotatebox[origin=c]{90}{ joint } ")
    for k in range(0, 5):
        name = ' & \\includegraphics[width=0.175\\textwidth]{{fig1/beta_joint_noise'+np.str(k > 0)+'_ang'+str(len(theta))+'_shift'+str(shift) + \
            '_maxint'+str(maxinta[k])+'}.png}'
        fid.write(name)
    fid.write("\\\\ \n")

    # delta
    fid.write(
        "\\multirow{2}{*}{\\rotatebox[origin=c]{90}{delta reconstruction\\hspace*{1cm} }}")
    # standard
    fid.write(" & \\rotatebox[origin=c]{90}{ standard } ")
    for k in range(0, 5):
        name = ' & \\includegraphics[width=0.175\\textwidth]{{fig1/delta_st_noise'+np.str(k > 0)+'_ang'+str(len(theta))+'_shift'+str(shift) + \
            '_maxint'+str(maxinta[k])+'}.png}'
        fid.write(name)
    fid.write("\\\\ \n")
    # joint 0
    fid.write("& \\rotatebox[origin=c]{90}{ coupled } ")
    for k in range(0, 5):
        name = ' & \\includegraphics[width=0.175\\textwidth]{{fig1/delta_joint0_noise'+np.str(k > 0)+'_ang'+str(len(theta))+'_shift'+str(shift) + \
            '_maxint'+str(maxinta[k])+'}.png}'
        fid.write(name)
    fid.write("\\\\ \n")
    # joint
    fid.write("& \\rotatebox[origin=c]{90}{ joint } ")
    for k in range(0, 5):
        name = ' & \\includegraphics[width=0.175\\textwidth]{{fig1/delta_joint_noise'+np.str(k > 0)+'_ang'+str(len(theta))+'_shift'+str(shift) + \
            '_maxint'+str(maxinta[k])+'}.png}'
        fid.write(name)
    fid.write("\\\\ \n")

    fid.write("\\end{tabular}\n \\caption{Reconstruction. obj size=(%d,%d,%d), voxelsize=%.1e, prb size = %d, ang=%d, shift=%d, joint (piter,titer,NITER)=(%d,%d,%d), standard (piter,titer,NITER)=(%d,%d,%d)} \n \\end{figure*} " %
              (*obj.shape, voxelsize, prbsize, len(theta), shift, piter, titer, NITER, piters, titers, NITERs))

    fid.close()
