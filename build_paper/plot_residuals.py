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
        name = "res2/res_"+str(piter)+"_"+str(titer)+"_"+str(NITER)
        res = np.load(name+'.npy')    
        print("free & %d/%d & %d & %.1e & %.1e & %.1e & %.1e \\\\ \\hline" % (piter, titer, NITER,
                                                                             res[-1, 0], res[-1, 1], res[-1, 2], res[-1, 3]))

    #xdiff
    for k in range(len(pitera)):
        piter = pitera[k]
        titer = titera[k]
        NITER = NITERa[k]
        namex = "xdiff2/xdiff_"+str(piter)+"_"+str(titer)+"_"+str(NITER)
        xdiff= np.load(namex+'.npy')
        name = "res2/res_"+str(piter)+"_"+str(titer)+"_"+str(NITER)
        res = np.load(name+'.npy')    
        plt.plot(range(1,len(xdiff)+1),xdiff)
        plt.xlim(1,len(xdiff)+1)
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
        name = "res2/res_"+str(piter)+"_"+str(titer)+"_"+str(NITER)
        res = np.load(name+'.npy')[:,0]
        plt.plot(range(1,len(res)+1),res,label=str(piter)+"/"+str(titer))
        plt.xlim(1,len(res)+1)
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
        name = "res2/res_"+str(piter)+"_"+str(titer)+"_"+str(NITER)
        res = np.load(name+'.npy')[:,1]
        plt.plot(range(1,len(res)+1),res,label=str(piter)+"/"+str(titer))        
        plt.xlim(1,len(res)+1)
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
        name = "res2/res_noise_"+str(piter)+"_"+str(titer)+"_"+str(NITER)
        res = np.load(name+'.npy')    
        print("poisson & %d/%d & %d & %.1e & %.1e & %.1e & %.1e \\\\ \\hline " % (piter, titer, NITER,
                                                                             res[-1, 0], res[-1, 1], res[-1, 2], res[-1, 3]))
    #xdiff
    for k in range(len(pitera)):
        piter = pitera[k]
        titer = titera[k]
        NITER = NITERa[k]
        namex = "xdiff2/xdiff_noise_"+str(piter)+"_"+str(titer)+"_"+str(NITER)
        xdiff= np.load(namex+'.npy')
        plt.plot(range(1,len(xdiff)+1),xdiff)
        plt.xlim(1,len(xdiff)+1)
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
        name = "res2/res_noise_"+str(piter)+"_"+str(titer)+"_"+str(NITER)
        res = np.load(name+'.npy')[:,0]
        plt.plot(range(1,len(res)+1),res,label=str(piter)+"/"+str(titer))
        plt.xlim(1,len(res)+1)
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
        name = "res2/res_noise_"+str(piter)+"_"+str(titer)+"_"+str(NITER)
        res = np.load(name+'.npy')[:,1]
        plt.plot(range(1,len(res)+1),res,label=str(piter)+"/"+str(titer))
        plt.xlim(1,len(res)+1)
        plt.xlabel('outer iterations', fontsize=14, **csfont)
        plt.ylabel('residual norm', fontsize=14, **csfont)
        plt.yscale('log')        
        plt.title('Dual residual', fontsize=16, **csfont)
        plt.grid()        
    plt.legend(loc='upper right')        
    plt.savefig(name+'_dual_res.png')              
    plt.clf()
    