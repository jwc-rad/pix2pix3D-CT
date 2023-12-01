import os
import numpy as np
import glob
from io import StringIO
import pydicom
import csv
from tqdm import tqdm
import pandas as pd
from pix2pix3DCT.source.data_loader import MyDataLoader, WND, rWND
from pix2pix3DCT.source.my3dpix2pix2 import My3dPix2Pix

def my_dicoms_to_dataframe(basedir, cts):
    caselist = [os.path.join(basedir, x) for x in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, x))]
    file_list = []
    for x in cts:
        file_list.extend(glob.glob(os.path.join(basedir, '*/'+x+'/*.*')))

    tdcmpath = os.path.join(caselist[0], cts[0])
    tdcmpath = [os.path.join(tdcmpath, x) for x in os.listdir(tdcmpath) if x.lower().endswith('.dcm')][0]
    tdcm = pydicom.dcmread(tdcmpath)

    headers = []
    headers.append('filepath')

    for x in tdcm:
        if x.name == 'Pixel Data':
            continue
        elif 'Overlay' in x.name or 'Referring' in x.name or 'Acquisition' in x.name:
            continue
        else:
            name = x.name.replace(' ', '')
            headers.append(name)

    output = StringIO()
    csv_writer = csv.DictWriter(output, fieldnames=headers)
    csv_writer.writeheader()

    for f in tqdm(file_list):
        file = pydicom.dcmread(f)

        row = {}
        for x in file:
            row['filepath'] = f
            if x.name == 'Pixel Data':
                continue
            elif 'Overlay' in x.name or 'Referring' in x.name or 'Acquisition' in x.name:
                continue
            else:
                name = x.name.replace(' ', '')
                row[name] = x.value
        unwanted = set(row) - set(headers)
        for unwanted_key in unwanted: del row[unwanted_key]
        csv_writer.writerow(row)

    output.seek(0) # we need to get back to the start of the StringIO
    df = pd.read_csv(output)

    df['pid'] = df['filepath'].apply(lambda x: x.split(os.sep)[-3])
    df['ct'] = df['filepath'].apply(lambda x: x.split(os.sep)[-2])
    df['zpos'] = df['ImagePosition(Patient)'].apply(lambda x: x.split( )[-1])
    df['zpos'] = df['zpos'].str.strip(']')

    cols = df.columns.tolist()
    cols = cols[-3:] + cols[:-3]
    df = df[cols]

    df.to_feather(os.path.join(basedir, 'headers.ftr'))
    return df


def loop_over_case(gan, case, savedir, notruth=False):
    pid, zs = case
    
    dcm_A, dcm_B = gan.data_loader.load_dicoms(pid, (0,zs+1))
    if notruth:
        dcm_A = np.zeros(dcm_B.shape, dtype=dcm_B.dtype)
    
    a = []
    b = []
    for w in gan.data_loader.window2:
        a.append(WND(dcm_A,w))
    for w in gan.data_loader.window1:
        b.append(WND(dcm_B,w))  
    tot_A = np.stack(a, axis=-1)
    tot_B = np.stack(b, axis=-1)
    tot_A = tot_A.astype('float32')/127.5 - 1.
    tot_B = tot_B.astype('float32')/127.5 - 1.
    
    fakes_raw = np.full((gan.img_rows,gan.img_cols,zs),0,dtype=tot_B.dtype)
    counts_raw = np.full((gan.img_rows,gan.img_cols,zs),0,dtype=int)
    
    for i in tqdm_notebook(range(zs+1-gan.depth)):
        imgs_B = np.expand_dims(tot_B[:,:,i:i+gan.depth,:], axis=0)
        fake_A = gan.generator.predict(imgs_B)
        fake_A = 0.5 * fake_A + 0.5
        fake_A = rWND(255.*fake_A[:,:,:,:,0], gan.data_loader.window2[0])
    
        fakes_raw[:,:,i:i+gan.depth] += fake_A[0]
        counts_raw[:,:,i:i+gan.depth] += 1
    
    mcounts = counts_raw.copy()
    mcounts[mcounts==0] = 1
    fakes = np.divide(fakes_raw, mcounts)
    
    # random sample
    sample = np.random.choice(fakes.shape[-1])
    sample = np.stack((
        dcm_B[:,:,sample].astype(fakes.dtype),
        fakes[:,:,sample],
        dcm_A[:,:,sample].astype(fakes.dtype)
    ), axis=-1)
    
    df1 = gan.data_loader.df
    dcms1 = df1[(df1['pid']==pid)&(df1['ct']==gan.data_loader.cts[0])]['filepath'].tolist()
    
    newpath = os.path.join(savedir, pid)
    if not os.path.isdir(newpath):
        os.mkdir(newpath)
    newpath = os.path.join(newpath, 'dicom')
    if not os.path.isdir(newpath):
        os.mkdir(newpath)
    
    for N, y in tqdm_notebook(enumerate(dcms1)):
        x = fakes[:,:,N]
        ds = pydicom.dcmread(y)
    
        x = (x-float(ds.RescaleIntercept))/float(ds.RescaleSlope)
    
        x = x.astype('int16')
    
        ds.PixelData = x.tobytes()
    
        ds.SeriesNumber += 99000
        ds.SOPInstanceUID += '.99'
    
        newfile = os.path.join(newpath, os.path.basename(y))
        ds.save_as(newfile)
    
    return sample