import ast
import itertools
import numbers
import numpy as np
import os
import pandas as pd
import pickle
import pydicom
from skimage import exposure

def WND(X, W):
    R = 255.*(X-W[1]+0.5*W[0])/W[0]
    R[R<0] = 0
    R[R>255] = 255
    return R

def rWND(X, W):
    R = X/255.*W[0]+(W[1]-0.5*W[0])
    return R

# loads from dataframe

class MyDataLoader():
    def __init__(self, df, cts=('VNC','DE'), img_shape=(512,512,8), grid=(1,1,1),
                  window1=[(2048,0)], window2=[(2048,0)], rescale_intensity=False, splitvar=0.8 
                ):
        self.cts = cts
        self.img_shape = img_shape
        self.grid = grid
        self.window1 = window1
        self.window2 = window2
        self.rescale_intensity = rescale_intensity
        self.splitvar = splitvar
        
        # dicom dataframe
        def slice_count(x):
            x['xx'] = x['ImagePosition(Patient)'].apply(lambda x: x.split( )[-1])
            x['xx'] = x['xx'].str.strip(']').astype(float)
            M = x['xx'].max()
            m = x['xx'].min()
            x['slice_pos'] = (M-x['xx'])/(M-m)
            x['slice_num'] = x['xx'].rank(method='min', ascending=False).astype(int)-1
            del x['xx']
            return x

        self.df=slice_count(df)
        self.df = self.df.groupby(['pid', 'ct']).apply(slice_count).reset_index(drop=True)
        self.df = self.df.sort_values(by=['pid','ct','slice_num']).reset_index(drop=True)
        tpid, tfilepath, self.rows, self.cols, self.rescale_in, self.rescale_sl = self.df.iloc[0][['pid', 'filepath','Rows','Columns','RescaleIntercept','RescaleSlope']]
        tsplit = tfilepath.split(os.sep)
        self.basedir = os.path.join(*tsplit[:tsplit.index(tpid)])
        
        qstring = 'ct=="'+self.cts[0]+'"'
        dff = self.df.query(qstring).groupby('pid')['slice_num'].max()+1
        self.case_list = [(k,dff[k]) for k in dff.index.tolist()]
                
        # split train/non-train sets
        self.case_split = None
        self.split()
        
        # get total_samples
        self.total_samples = self.get_total_samples()
                
    def split(self):
        N = len(self.case_list)
        s = np.full(N, False)
        
        # random split or load from split.pkl
        if isinstance(self.splitvar, numbers.Number):
            choose = np.random.choice(N, size=int(self.splitvar*N), replace=False)
            s[choose] = True
            
            self.case_split = []
            self.case_split.append(list(itertools.compress(self.case_list, s)))
            self.case_split.append(list(itertools.compress(self.case_list, ~s)))
        else:
            with open(self.splitvar, 'rb') as f:
                self.case_split = pickle.load(f)
            
    def save_split(self, savepath):
        with open(savepath, 'wb') as f:
            pickle.dump(self.case_split, f)
            
    def get_total_samples(self):
        A = []
        gr, gc, gz = self.grid
        
        for c in self.case_split:
            C = np.array([], dtype='uint16').reshape(0,4)
            
            for i, case in enumerate(c):
                _, tz = case
                
                x,y,z = np.meshgrid(
                    range(1+(self.rows-self.img_shape[0])//gr), 
                    range(1+(self.cols-self.img_shape[1])//gc), 
                    range(1+(tz-self.img_shape[2])//gz)
                )
                B = np.moveaxis(np.array([gr*x,gc*y,gz*z]),0,-1).reshape(-1,3).astype('uint16')
                k = np.full((B.shape[0],1),i,dtype='uint16')
                B = np.concatenate((k,B), axis=-1)
                C = np.concatenate((C,B), axis=0)
            A.append(C)
        return A
    
    def load_dicoms(self, pid, slice_nums, window=False):
        if isinstance(pid,int):
            pid = self.case_list[pid][0]
        slice_num_start, slice_num_end = slice_nums        
        
        querystring = 'slice_num>='+str(slice_num_start)+'&slice_num<'+str(slice_num_end)+'&pid=="'+pid+'"'
        dA = self.df.query(querystring)[['ct','filepath']]

        vols = []
        for i in range(2):
            vol = []
            for x in dA.query('ct=="'+self.cts[i]+'"')['filepath']:
                vol.append(pydicom.dcmread(x).pixel_array)
            vols.append(np.array(vol).astype(float)*self.rescale_sl + self.rescale_in)

        A = np.moveaxis(np.array(vols[1]), 0, -1)
        B = np.moveaxis(np.array(vols[0]), 0, -1)
        
        if window:
            A = WND(A,self.window2[0])
            B = WND(B,self.window1[0])
        
        return A, B
    
    def imread(self, case, pos, window=True, split=0):
        rx, ry, rz = pos
        pid, zs = self.case_split[split][case]

        xm = rx
        xM = xm + self.img_shape[1]
        ym = ry
        yM = ym + self.img_shape[0]

        slice_num_start = rz
        slice_num_end = rz+self.img_shape[2]

        A, B = self.load_dicoms(pid,(slice_num_start,slice_num_end))
        A = A[ym:yM,xm:xM,:]
        B = B[ym:yM,xm:xM,:]

        if window:
            a = []
            b = []
            for w in self.window2:
                a.append(WND(A,w))
            for w in self.window1:
                b.append(WND(B,w))  
            A = np.stack(a, axis=-1)
            B = np.stack(b, axis=-1)

        return A, B
    
    def imread_slice(self, case, pos, window=True, split=0):
        rx, ry, rz = pos
        pid, zs = self.case_split[split][case]

        xm = rx
        xM = xm + self.img_shape[1]
        ym = ry
        yM = ym + self.img_shape[0]

        slice_num_start = rz
        slice_num_end = rz+self.img_shape[2]

        A, B = self.load_dicoms(pid,(slice_num_start,slice_num_end))
        A = A[ym:yM,xm:xM,:]
        B = B[ym:yM,xm:xM,:]

        if window:
            a = []
            b = []
            for w in self.window2:
                a.append(WND(A,w))
            for w in self.window1:
                b.append(WND(B,w))  
            A = np.stack(a, axis=-1)
            B = np.stack(b, axis=-1)           

        return A, B
        
    def load_batch(self, batch_size=1, split=0):
        total_samples = self.total_samples[split]
        self.n_batches = int(len(total_samples) / batch_size)
        
        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        
        points = np.random.choice(len(total_samples), size=self.n_batches*batch_size, replace=False)

        for i in range(self.n_batches):
            batch_p = points[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for p in batch_p:
                c = total_samples[p][0]
                pos = total_samples[p][1:]
                
                img_A, img_B = self.imread(c, pos, split=split)
                if self.rescale_intensity:
                    # rescale to -0.95~0.95
                    img_A = exposure.rescale_intensity(img_A, out_range=(-0.95,0.95))
                    img_B = exposure.rescale_intensity(img_B, out_range=(-0.95,0.95))
                else:
                    img_A = img_A/127.5 - 1.
                    img_B = img_B/127.5 - 1.
                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)
            imgs_B = np.array(imgs_B)

            yield imgs_A, imgs_B    
            
    def load_data(self, batch_size=1, split=0):
        total_samples = self.total_samples[split]
        batches = np.random.choice(len(total_samples), size=batch_size, replace=False)
        
        imgs_A = []
        imgs_B = []
        for b in batches:
            c = total_samples[b][0]
            pos = total_samples[b][1:]
            img_A, img_B = self.imread(c, pos, split=split)
            imgs_A.append(img_A)
            imgs_B.append(img_B)
            
        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.
               
        return imgs_A, imgs_B

   
