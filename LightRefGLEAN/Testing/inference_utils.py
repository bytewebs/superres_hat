import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))
sys.path.append(str(Path('.').absolute().parent))

from osgeo import osr, gdal
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from config import config
import torch.optim as optim
from sensor import Sensor
from torch import nn
from torch.utils.data import DataLoader
from skimage.transform import resize
import torch.nn.functional as F

from utils import normalise_bandwise_RGB, recover_bandwise_RGB, normalise_bandwise_20m, recover_bandwise_20m
from tqdm import tqdm
from skimage.exposure import match_histograms


# config
GDAL_DATA_TYPE = gdal.GDT_Float64 
GEOTIFF_DRIVER_NAME = "GTiff"
NO_DATA = -32767
SPATIAL_REFERENCE_SYSTEM_WKID = 32767

def create_raster(output_path,
                  columns,
                  rows,
                  nband = 1,
                  gdal_data_type = GDAL_DATA_TYPE,
                  driver = GEOTIFF_DRIVER_NAME):

    driver = gdal.GetDriverByName(driver)

    output_raster = driver.Create(output_path, int(columns), int(rows), nband, eType = gdal_data_type)    
    return output_raster

def numpy_array_to_raster(output_path,
                          numpy_array,
                          proj,
                          trans,
                          nband = 1,
                          no_data = NO_DATA,
                          gdal_data_type = GDAL_DATA_TYPE,
                          spatial_reference_system_wkid = SPATIAL_REFERENCE_SYSTEM_WKID,
                          driver = GEOTIFF_DRIVER_NAME):


    shp = numpy_array.shape
    rows, columns = shp[1], shp[2]
    print(rows, columns)
    output_raster = create_raster(output_path, int(columns), int(rows), nband, gdal_data_type) 
    geotransform = trans
    print(geotransform)

    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromWkt(proj)
    # spatial_reference.ImportFromProj4('+proj=lcc +lat_1=15 +lat_2=65 +lat_0=30 +lon_0=95 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs')
    output_raster.SetProjection(spatial_reference.ExportToWkt())
    output_raster.SetGeoTransform(geotransform)
    for i in range(1, nband+1):
        print(i)
        output_band = output_raster.GetRasterBand(i)
        output_band.SetNoDataValue(no_data)
        output_band.WriteArray(numpy_array[i-1])          
        output_band.FlushCache()
#         print(output_raster.GetProjection())
    
    return  output_raster


def generate_raster_all(file_name, chpt_rgb, chpt, raster_file='../sentinel_clip.tif', scale=4, mscale=4, pad=5, model='esrgan'):
    s = Sensor('S2')
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # Creating generator model
    # Creating generator model
    if model=='edsr':
        from networks.model_test.edsr import EDSR
        netG_rgb = EDSR().to(device)
        netG = EDSR(scale=4).to(device)
    elif model=='esrgan':
        from networks import RRDBNet
        netG_rgb = RRDBNet(in_channels=4, out_channels=4, channels=config.NETWORK.G_CHANNELS, 
                        growth_channels=config.NETWORK.GROWTH_CHANNELS, num_rrdb=13,  
                        upscale=2).to(device)
        netG = RRDBNet(in_channels=4, out_channels=4, channels=config.NETWORK.G_CHANNELS, 
                        growth_channels=config.NETWORK.GROWTH_CHANNELS, num_rrdb=config.NETWORK.NUM_RRDB,  
                        upscale=s.ratio).to(device)
    elif model=='rcan':
        from networks.model_test.rcan import RCAN
        from networks.model_test.edsr import EDSR
        netG_rgb = RCAN().to(device)
        netG = EDSR(scale=4).to(device)
    
    netG_rgb = nn.DataParallel(netG_rgb, device_ids=[1, 0, 2, 3])
    netG = nn.DataParallel(netG, device_ids=[1, 0, 2, 3])
    
    netG_rgb.load_state_dict(torch.load(chpt_rgb))
    netG.load_state_dict(torch.load(chpt))

    ds = gdal.Open(raster_file)
    arr = ds.ReadAsArray()
    proj = ds.GetProjection()
    geo = ds.GetGeoTransform()
    geo2 = np.array(geo)
    geo2[1] = geo[1]/scale
    geo2[-1] = geo[-1]/scale
    ds = None

    dsarr = arr.copy()
    b24 = torch.FloatTensor(arr[:4])
    b47 = torch.FloatTensor(arr[4:])
    b = b47 #torch.cat((b47, b8a), 0)
    patch_sz = 118

    ms_norm = []
    for i in range(arr.shape[0]):
        i, j = np.percentile(arr[i], [1, 99])
        ms_norm.append([max(0, i-100), j+100])
    ms_norm = np.array(ms_norm)
    print(ms_norm.shape, ms_norm)
    
    b_rgb = normalise_bandwise_RGB(b24, ms_norm).unsqueeze(0)   # Normalising entire raster
    b_rgb = torch.clip(b_rgb, 0, 1)
    
    b = normalise_bandwise_20m(b.clone(), ms_norm).unsqueeze(0)   # Normalising entire raster
    b = torch.clip(b, 0, 1)
    
    b_comb = torch.cat((b_rgb, b), 1)[0].numpy()
    ds = numpy_array_to_raster(file_name.replace('.tif', '_normalised.tif'), (b_comb), proj, geo, nband=8)
    ds.FlushCache()
    ds = None
    
    _, _, hb, wb = b.shape
    bn = torch.zeros(1, 4, (((hb//patch_sz)+1)*patch_sz), (((wb//patch_sz)+1)*patch_sz))
    bn[:,:,:hb,:wb] = b
    
    bn_rgb = torch.zeros(1, 4, (((hb//patch_sz)+1)*patch_sz), (((wb//patch_sz)+1)*patch_sz))
    bn_rgb[:,:,:hb,:wb] = b_rgb
    
    _, _, h, w = bn.shape
    
    patches = torch.zeros((h//patch_sz)*(w//patch_sz), 4, patch_sz, patch_sz)
    patches_rgb = torch.zeros((h//patch_sz)*(w//patch_sz), 4, patch_sz, patch_sz)
    patches_out = np.zeros(((h//patch_sz)*(w//patch_sz), 8, patch_sz*scale, patch_sz*scale))
    num_patch = patches.shape[0]
    
    # Save Chops of original image
    i = 0
    _, _, h, w = b.shape
    for x0 in range(0, h, patch_sz):
        for y0 in range(0, w, patch_sz):
            patches[i] = bn[:,:,x0:x0+patch_sz,y0:y0+patch_sz]
            patches_rgb[i] = bn_rgb[:,:,x0:x0+patch_sz,y0:y0+patch_sz]
            i+=1
    
    patches_pad = F.pad(input=patches, pad=(pad, pad, pad, pad), mode='reflect')
    patches_rgb_pad = F.pad(input=patches_rgb, pad=(pad, pad, pad, pad), mode='reflect')
    
    for i in tqdm(range(patches.shape[0])):
        arr = patches_pad[i:i+1,:,::2,::2]
        arr_rgb = patches_rgb_pad[i:i+1]
        o_rgb = netG_rgb(arr_rgb.to('cuda:3'))
#         o_rgb = F.interpolate(arr_rgb.to('cuda:3'), scale_factor=2, mode="nearest")
        o = netG(arr.to('cuda:3'))
        out = np.moveaxis(np.array(o[0].detach().cpu()), 0, -1)
        out = out[(mscale*pad):-(mscale*pad),(mscale*pad):-(mscale*pad)]
        out = np.moveaxis(resize(out, (patch_sz*scale, patch_sz*scale)), -1, 0)
        
        out_rgb = np.moveaxis(np.array(o_rgb[0].detach().cpu()), 0, -1)
        out_rgb = out_rgb[(mscale*pad):-(mscale*pad),(mscale*pad):-(mscale*pad)]
        out_rgb = np.moveaxis(resize(out_rgb, (patch_sz*scale, patch_sz*scale)), -1, 0)
        
        patches_out[i,:4] = out_rgb
        patches_out[i,4:] = out
        o_rgb = o_rgb.detach().cpu()
        torch.cuda.empty_cache()
        
    _, _, h, w = bn.shape
    out_raster = np.zeros((1, 8, h*scale, w*scale))
    i=0
    for x0 in range(0, h*scale, patch_sz*scale):
        for y0 in range(0, w*scale, patch_sz*scale):
            out_raster[:,:,x0:x0+(patch_sz*scale),y0:y0+(patch_sz*scale)] = patches_out[i]
            i+=1

    out_raster = out_raster[:,:,:scale*hb,:scale*wb].copy()

    out_raster_m = out_raster[0].copy()
    out_raster_m[:4] = recover_bandwise_RGB(out_raster[:,:4], ms_norm)
    out_raster_m[4:] = recover_bandwise_20m(out_raster[:,4:], ms_norm)
    
    out_raster_m = match_histograms(out_raster_m, dsarr[:], channel_axis=0)
    ds = numpy_array_to_raster(file_name, (out_raster_m), proj, geo2, nband=8)
    ds.FlushCache()
    ds = None
    

def generate_raster_glean(file_name, chpt_rgb, chpt, raster_file='../sentinel_clip.tif', scale=4, mscale=4, pad=5, rgb_model='esrgan', model='glean'):
    s = Sensor('S2')
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # Creating generator model
    if rgb_model=='esrgan':
        from networks import RRDBNet
    elif rgb_model=='esrgan_pa':
        from networks import RRDBNet2 as RRDBNet
    elif rgb_model=='esrgan_ca':
        from networks import RRDBNet3 as RRDBNet
    
    if mscale==4:
        netG_rgb = RRDBNet(in_channels=4, out_channels=4, channels=config.NETWORK.G_CHANNELS, 
                    growth_channels=config.NETWORK.GROWTH_CHANNELS, num_rrdb=config.NETWORK.NUM_RRDB,  
                    upscale=s.ratio).to(device)
    elif mscale == 2:
        netG_rgb = RRDBNet(in_channels=4, out_channels=4, channels=config.NETWORK.G_CHANNELS, 
                growth_channels=config.NETWORK.GROWTH_CHANNELS, num_rrdb=13,  
                upscale=2).to(device)

    netG_rgb = nn.DataParallel(netG_rgb, device_ids=[1, 0, 2, 3])
    if model=='glean':
        from networks.glean import GLEANStyleGANv2
        netG = GLEANStyleGANv2(64, 256, 4, 64, 23, rgb_channels=4).to(device)
    elif model=='light_glean':
        from networks.glean import LightGLEANStyleGANv2
        netG = LightGLEANStyleGANv2(64, 256, 4, 64, 23, rgb_channels=4).to(device)
    netG = nn.DataParallel(netG, device_ids=[1, 0, 2, 3])
    netG_rgb.load_state_dict(torch.load(chpt_rgb))
    netG.load_state_dict(torch.load(chpt))
    
    ds = gdal.Open(raster_file)
    arr = ds.ReadAsArray()
    proj = ds.GetProjection()
    geo = ds.GetGeoTransform()
    geo2 = np.array(geo)
    geo2[1] = geo[1]/scale
    geo2[-1] = geo[-1]/scale
    ds = None
    
    dsarr = arr.copy()
    
    b24 = torch.FloatTensor(arr[:4])
    b47 = torch.FloatTensor(arr[4:])
    b = b47 #torch.cat((b47, b8a), 0)
    patch_sz = 118   # Variable argument
    
    ms_norm = []
    for i in range(arr.shape[0]):
        i, j = np.percentile(arr[i], [1, 99])
        ms_norm.append([max(0, i-100), j+100])
    ms_norm = np.array(ms_norm)
    print(ms_norm.shape, ms_norm)

    b_rgb = normalise_bandwise_RGB(b24, ms_norm).unsqueeze(0)   # Normalising entire raster
    b_rgb = torch.clip(b_rgb, 0, 1)
    
    b = normalise_bandwise_20m(b.clone(), ms_norm).unsqueeze(0)   # Normalising entire raster
    b = torch.clip(b, 0, 1)
    
    b_comb = torch.cat((b_rgb, b), 1)[0].numpy()
    ds = numpy_array_to_raster(file_name.replace('.tif', '_normalised.tif'), (b_comb), proj, geo, nband=8)
    ds.FlushCache()
    ds = None
    
    _, _, hb, wb = b.shape
    bn = torch.zeros(1, 4, (((hb//patch_sz)+1)*patch_sz), (((wb//patch_sz)+1)*patch_sz))
    bn[:,:,:hb,:wb] = b
    
    bn_rgb = torch.zeros(1, 4, (((hb//patch_sz)+1)*patch_sz), (((wb//patch_sz)+1)*patch_sz))
    bn_rgb[:,:,:hb,:wb] = b_rgb
    
    _, _, h, w = bn.shape
    
    patches = torch.zeros((h//patch_sz)*(w//patch_sz), 4, patch_sz, patch_sz)
    patches_rgb = torch.zeros((h//patch_sz)*(w//patch_sz), 4, patch_sz, patch_sz)
    patches_out = np.zeros(((h//patch_sz)*(w//patch_sz), 8, patch_sz*scale, patch_sz*scale))
    num_patch = patches.shape[0]
    
    # Save Chops of original image
    i = 0
    _, _, h, w = b.shape
    for x0 in range(0, h, patch_sz):
        for y0 in range(0, w, patch_sz):
            patches[i] = bn[:,:,x0:x0+patch_sz,y0:y0+patch_sz]
            patches_rgb[i] = bn_rgb[:,:,x0:x0+patch_sz,y0:y0+patch_sz]
            i+=1
    
    patches_pad = F.pad(input=patches, pad=(pad, pad, pad, pad), mode='reflect')
    patches_rgb_pad = F.pad(input=patches_rgb, pad=(pad, pad, pad, pad), mode='reflect')
    
    for i in tqdm(range(patches.shape[0])):
        arr = patches_pad[i:i+1,:,::2,::2]
        arr_rgb = patches_rgb_pad[i:i+1]
        o_rgb = netG_rgb(arr_rgb.to('cuda:3'))
        o = netG(arr.to('cuda:3'), o_rgb)
        out = np.moveaxis(np.array(o[0].detach().cpu()), 0, -1)
        out = out[(mscale*pad):-(mscale*pad),(mscale*pad):-(mscale*pad)]
        out = np.moveaxis(resize(out, (patch_sz*scale, patch_sz*scale)), -1, 0)
        
        out_rgb = np.moveaxis(np.array(o_rgb[0].detach().cpu()), 0, -1)
        out_rgb = out_rgb[(mscale*pad):-(mscale*pad),(mscale*pad):-(mscale*pad)]
        out_rgb = np.moveaxis(resize(out_rgb, (patch_sz*scale, patch_sz*scale)), -1, 0)
        
        patches_out[i,:4] = out_rgb
        patches_out[i,4:] = out
        o_rgb = o_rgb.detach().cpu()
        torch.cuda.empty_cache()
        
    _, _, h, w = bn.shape
    out_raster = np.zeros((1, 8, h*scale, w*scale))
    i=0
    for x0 in range(0, h*scale, patch_sz*scale):
        for y0 in range(0, w*scale, patch_sz*scale):
            out_raster[:,:,x0:x0+(patch_sz*scale),y0:y0+(patch_sz*scale)] = patches_out[i]
            i+=1

    out_raster = out_raster[:,:,:scale*hb,:scale*wb].copy()
#     print('out', out_raster.shape)
#     out_raster2 = recover_bandwise_20m(out_raster, ms_norm)[0]
#     print(np.mean(out_raster2, (1, 2)), out_raster2.shape, arr[4:].shape)
    
    out_raster_m = out_raster[0].copy()
    out_raster_m[:4] = recover_bandwise_RGB(out_raster[:,:4], ms_norm)
    out_raster_m[4:] = recover_bandwise_20m(out_raster[:,4:], ms_norm)
    
#     out_raster_m = match_histograms(out_raster2, dsarr[4:], channel_axis=0)
    out_raster_m = match_histograms(out_raster_m, dsarr[:], channel_axis=0)
    ds = numpy_array_to_raster(file_name, (out_raster_m), proj, geo2, nband=8)
    ds.FlushCache()
    ds = None
