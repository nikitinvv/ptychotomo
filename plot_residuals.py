import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # ADMM    
    pitera = [500, 500, 1, 200, 4, 1]
    titera = [500, 500, 200, 1, 4, 1]
    NITERa = [40, 200, 200, 200, 200, 200]
    csfont = {'fontname':'Times New Roman'}
    #table
    for k in range(len(pitera)):
        piter = pitera[k]
        titer = titera[k]
        NITER = NITERa[k]
        name = "res/res_"+str(piter)+"_"+str(titer)+"_"+str(NITER)
        res = np.load(name+'.npy')    
        print("noise free & %d/%d & %d & %.1e & %.1e & %.1e & %.1e \\\\ \n" % (piter, titer, NITER,
                                                                             res[-1, 0], res[-1, 1], res[-1, 2], res[-1, 3]))
        print(res[:,2])
    exit()        

    #xdiff
    for k in range(len(pitera)):
        piter = pitera[k]
        titer = titera[k]
        NITER = NITERa[k]
        namex = "xdiff/xdiff_"+str(piter)+"_"+str(titer)+"_"+str(NITER)
        xdiff= np.load(namex+'.npy')
        name = "res/res_"+str(piter)+"_"+str(titer)+"_"+str(NITER)
        res = np.load(name+'.npy')    
        plt.plot(xdiff)
        plt.xlabel('iterations', fontsize=14, **csfont)
        plt.ylabel('RMS error', fontsize=14, **csfont)
        plt.yscale('log')        
        plt.title('Tomography convergence',fontsize=16,**csfont)
        plt.grid()
        plt.savefig(namex+'.png')
        plt.clf()
        
    # p res
    for k in range(1,len(pitera)):
        piter = pitera[k]
        titer = titera[k]
        NITER = NITERa[k]
        name = "res/res_"+str(piter)+"_"+str(titer)+"_"+str(NITER)
        res = np.load(name+'.npy')[:,0]
        plt.plot(res,label=str(piter)+"/"+str(titer))
        plt.xlabel('outer iterations', fontsize=14, **csfont)
        plt.ylabel('residual norm', fontsize=14, **csfont)
        plt.yscale('log')        
        plt.title('Primal residual', fontsize=16, **csfont)
        plt.grid()        
    plt.legend(loc='upper right')        
    plt.savefig(name+'_primal_res.png')        
    plt.clf()
    
    # d res
    for k in range(1,len(pitera)):
        piter = pitera[k]
        titer = titera[k]
        NITER = NITERa[k]
        name = "res/res_"+str(piter)+"_"+str(titer)+"_"+str(NITER)
        res = np.load(name+'.npy')[:,1]
        plt.plot(res,label=str(piter)+"/"+str(titer))
        plt.xlabel('outer iterations', fontsize=14, **csfont)
        plt.ylabel('residual norm', fontsize=14, **csfont)
        plt.yscale('log')        
        plt.title('Dual residual', fontsize=16, **csfont)
        plt.grid()        
    plt.legend(loc='upper right')        
    plt.savefig(name+'_dual_res.png')              
    plt.clf()
    



###############Noise
    #table
    for k in range(len(pitera)):
        piter = pitera[k]
        titer = titera[k]
        NITER = NITERa[k]
        name = "res/res_noise_"+str(piter)+"_"+str(titer)+"_"+str(NITER)
        res = np.load(name+'.npy')    
        print("poisson noise & %d/%d & %d & %.1e & %.1e & %.1e & %.1e \\\\ \n" % (piter, titer, NITER,
                                                                             res[-1, 0], res[-1, 1], res[-1, 2], res[-1, 3]))
    #xdiff
    for k in range(len(pitera)):
        piter = pitera[k]
        titer = titera[k]
        NITER = NITERa[k]
        namex = "xdiff/xdiff_noise_"+str(piter)+"_"+str(titer)+"_"+str(NITER)
        xdiff= np.load(namex+'.npy')
        plt.plot(xdiff)
        plt.xlabel('iterations', fontsize=14, **csfont)
        plt.ylabel('RMS error', fontsize=14, **csfont)
        plt.yscale('log')        
        plt.title('Tomography convergence',fontsize=16,**csfont)
        plt.grid()
        plt.savefig(namex+'.png')
        plt.clf()
    # p res
    for k in range(1,len(pitera)):
        piter = pitera[k]
        titer = titera[k]
        NITER = NITERa[k]
        name = "res/res_noise_"+str(piter)+"_"+str(titer)+"_"+str(NITER)
        res = np.load(name+'.npy')[:,0]
        plt.plot(res,label=str(piter)+"/"+str(titer))
        plt.xlabel('outer iterations', fontsize=14, **csfont)
        plt.ylabel('residual norm', fontsize=14, **csfont)
        plt.yscale('log')        
        plt.title('Primal residual', fontsize=16, **csfont)
        plt.grid()        
    plt.legend(loc='upper right')        
    plt.savefig(name+'_primal_res.png')              
    plt.clf()
    # d res
    for k in range(1,len(pitera)):
        piter = pitera[k]
        titer = titera[k]
        NITER = NITERa[k]
        name = "res/res_noise_"+str(piter)+"_"+str(titer)+"_"+str(NITER)
        res = np.load(name+'.npy')[:,1]
        plt.plot(res,label=str(piter)+"/"+str(titer))
        plt.xlabel('outer iterations', fontsize=14, **csfont)
        plt.ylabel('residual norm', fontsize=14, **csfont)
        plt.yscale('log')        
        plt.title('Dual residual', fontsize=16, **csfont)
        plt.grid()        
    plt.legend(loc='upper right')        
    plt.savefig(name+'_dual_res.png')              
    plt.clf()
    