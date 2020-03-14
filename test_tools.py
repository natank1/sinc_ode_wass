

import numpy as np
th1=95000000.
# th1=9.5

tth0= 1./th1
class test_tools():
    def __init__(self,flag,flag_t,ldc=False):
        self.c0_m0=0
        self.c0_m1 = 0
        self.c1_m0 = 0
        self.c1_m1 = 0
        self.graph = flag
        self.ldc= ldc
        self.agg_reults =list()

        if flag_t==3:
            self.measure_res =self.no_proc
        else:
            if self.graph:
                self.measure_res =self.raw_res
            else:
                self.measure_res = self.new_res
        return
    def no_proc(self,out,m_labels):
        return

    def convert_to_zz(self,out,m_labels):
        zz = out.cpu().numpy()
        tt = m_labels.cpu().numpy()
        tt = np.expand_dims(tt, axis=1)
        zz = np.concatenate((zz, tt), axis=1)
        f1, f2 = zz.shape

        return zz,tt,f1,f2
    def raw_res(self,out,m_labels   ):
        zz,tt,f1,f2=self.convert_to_zz(out,m_labels)

        if self.ldc:
            m1 = np.exp(zz[:,:2])
            ms =m1.sum(axis=1)
            ms =np.expand_dims(ms,axis=1)
            m0= m1/ms
            zz = np.concatenate((m0, tt), axis=1)

        # ms = m1.sum()
        # zz = np.array([m1[0] / ms, m1[1] / ms, zz[2]])
        # print (zz)



        for j in range(f1):

            if ( tth0*zz[j, 0] < zz[j, 1]) and (zz[j, 2] == 1) :
                self.c1_m1 += 1
            if (tth0*zz[j, 0] < zz[j, 1]) and (zz[j, 2] == 0):
                self.c0_m1+= 1
            if (tth0*zz[j, 0] > zz[j, 1]) and (zz[j, 2] == 1):
                self.c1_m0 += 1
            if (tth0*zz[j, 0] > zz[j, 1]) and (zz[j, 2] == 0):
                self.c0_m0 += 1

    def new_res(self,out,m_labels):
        zz, tt,f1, f2 = self.convert_to_zz(out, m_labels)

        for j in range(f1):
            self.agg_reults.append(zz[j,:])
            if ( tth0*zz[j, 0] < zz[j, 1]) and (zz[j, 2] == 1):
                self.c1_m1 += 1
            if ( tth0*zz[j, 0] < zz[j, 1]) and (zz[j, 2] == 0):
                self.c0_m1+= 1
            if ( tth0*zz[j, 0] > zz[j, 1]) and (zz[j, 2] == 1):
                self.c1_m0 += 1
            if ( tth0*zz[j, 0] > zz[j, 1]) and (zz[j, 2] == 0):
                self.c0_m0 += 1

    def final_score(self):
        print ("here it is")
        kk=[]
        for i in self.agg_reults:
            m1= np.exp(i[:2])
            ms =m1.sum()
            vec0= np.array([m1[0]/ms,m1[1]/ms,i[2]])
            print (vec0[:2].sum(),vec0[2])
            kk.append(vec0)
        kk= np.array(kk)
        np.save("graph",kk)




