

import numpy as np
import cv2
import matplotlib.pyplot as plt
from osgeo import gdal, gdalconst
import math
from skimage import measure, color, morphology
from skimage import feature
from skimage import filters
from skimage import exposure
import geopandas as gpd
import pandas as pd
import os
import time
import scipy
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import joblib

# Read image
def ReadTif(filepath):
    dataset = gdal.Open(filepath)
    im_bands = dataset.RasterCount
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    geotransform = dataset.GetGeoTransform()
    im_projection = dataset.GetProjection()
    if geotransform[5]>0:
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
        im_data = np.flipud(im_data)
        geotransform = (
            geotransform[0],  # 左上角 x 坐标
            geotransform[1],  # x 方向像元大小
            geotransform[2],  # x 方向旋转参数
            geotransform[3]+ geotransform[5] * im_height,  # 左上角 y 坐标
            geotransform[4],  # y 方向旋转参数并取反
            -geotransform[5]  # y 方向像元大小
            )
    else:
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    return im_data, im_width,im_height,im_bands,geotransform,im_projection


## Save image
def Savetiff(im_data, im_width, im_height, im_bands, im_geotrans, im_proj, filepath):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filepath, im_width, im_height, im_bands, datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


# Normalized
def normalized(imdata, im_height, im_width):
    i = 0
    img_nor = np.zeros([im_height, im_width])
    while i < im_height:
        j = 0
        while j < im_width:
            if imdata[i,j] >= 500:
                img_nor[i,j] = 1
            elif imdata[i,j] <= 200:
                img_nor[i, j] = 0
            else:
                img_nor[i,j] = (imdata[i,j]-200)/300
            j=j+1
        i=i+1
    return img_nor


# Water mapping
def waterextract(img):
    # 0-255
    img = img * 255
    img_data = np.uint8(img)
    # Otsu extract water
    ret1, Otsu = cv2.threshold(img_data, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    Water_extent = cv2.bitwise_and(img_data, Otsu)
    # Morphological operation;
    kernel = np.ones((5, 5), np.uint8)
    can_region = cv2.dilate(Water_extent, kernel)
    # Adaptive threshold
    im_AT = cv2.bitwise_and(img_data, can_region)
    AT_water = cv2.adaptiveThreshold(im_AT, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                cv2.THRESH_BINARY, 7, 5 )
    water = cv2.bitwise_and(AT_water, can_region)
    return water






## Register the VV and VH

def get_intersection(VH_path,VV_path,S2_path, DEM_path, Slope_path,label_path):
    # 打开VV,VH影像
    VH = gdal.Open(VH_path)
    VV = gdal.Open(VV_path)
    
    ##打开S2
    S2 = gdal.Open(S2_path)
    NDWI = S2.GetRasterBand(1)
    NDVI = S2.GetRasterBand(2)
    ##打开DEM
    DEM = gdal.Open(DEM_path)
    slope = gdal.Open(Slope_path)
    # 打开label影像
    label = gdal.Open(label_path)
    
    # 获取VH影像的范围
    Flip_label_VH = False
    VH_geo_transform = VH.GetGeoTransform()
    if VH_geo_transform[5]>0:
        VH_geo_transform = list(VH_geo_transform)
        VH_y_max = VH_geo_transform[3]+VH_geo_transform[5] * VH.RasterYSize
        VH_y_min = VH_geo_transform[3]
        VH_geo_transform[5] = -VH_geo_transform[5]
        Flip_label_VH = True
    else:
        VH_y_max = VH_geo_transform[3]
        VH_y_min = VH_y_max + VH_geo_transform[5] * VH.RasterYSize
    VH_x_min = VH_geo_transform[0]
    VH_x_max = VH_x_min + VH_geo_transform[1] * VH.RasterXSize
    
    # 获取S2影像的范围
    Flip_S2 = False
    S2_geo_transform = S2.GetGeoTransform()
    if S2_geo_transform[5]>0:
        Flip_S2 = True
        
    # 获取DEM影像的范围
    Flip_DEM = False
    DEM_geo_transform = DEM.GetGeoTransform()
    if DEM_geo_transform[5]>0:
        Flip_DEM = True
        
    # 获取DEM影像的范围
    Flip_slope = False
    DEM_geo_transform = DEM.GetGeoTransform()
    if DEM_geo_transform[5]>0:
        Flip_DEM = True
    
    # 获取slope影像的范围
    Flip_slope = False
    slope_geo_transform = slope.GetGeoTransform()
    if slope_geo_transform[5]>0:
        Flip_slope = True
    
    # 获取slope影像的范围
    Flip_label_VV = False
    VV_geo_transform = VV.GetGeoTransform()
    if VV_geo_transform[5]>0:
        Flip_label_VV = True
    
    # 获取label的范围
    label_geo_transform = label.GetGeoTransform()
    label_x_min = label_geo_transform[0]
    label_y_max = label_geo_transform[3]
    label_x_max = label_x_min + label_geo_transform[1] * label.RasterXSize
    label_y_min = label_y_max + label_geo_transform[5] * label.RasterYSize
    
    # 计算两张影像相交的范围
    x_min = max(VH_x_min, label_x_min)
    y_max = min(VH_y_max, label_y_max)
    x_max = min(VH_x_max, label_x_max)
    y_min = max(VH_y_min, label_y_min)
    
    # 计算相交区域的宽度和高度
    width = int((x_max - x_min) / VH_geo_transform[1])+1
    height = -int((y_max - y_min) / VH_geo_transform[5])+1
    
    ## offset
    col = int((x_min - VH_x_min) / VH_geo_transform[1])
    row = int((y_max - VH_y_max) / VH_geo_transform[5])
    
    # 读取相交区域的影像数据
    if Flip_label_VH:
        VH_data = np.flipud(VH.ReadAsArray(col,row,width,height))
    else:
        VH_data = VH.ReadAsArray(col,row,width,height)
    
    if Flip_label_VV:
        VV_data = np.flipud(VV.ReadAsArray(col,row,width,height))
    else:
        VV_data = VV.ReadAsArray(col,row,width,height)
    
    if Flip_S2:
        NDWI_data = np.flipud(NDWI.ReadAsArray(col,row,width,height))
        NDVI_data = np.flipud(NDVI.ReadAsArray(col,row,width,height))
    else:
        NDWI_data = NDWI.ReadAsArray(col,row,width,height)
        NDVI_data = NDVI.ReadAsArray(col,row,width,height)
    
    if Flip_DEM:
        DEM_band = DEM.GetRasterBand(1)
        DEM_data = np.flipud(DEM_band.ReadAsArray(col,row,width,height))
    else:
        DEM_band = DEM.GetRasterBand(1)
        DEM_data = DEM_band.ReadAsArray(col,row,width,height)
        
    if Flip_slope:
        slope_band = slope.GetRasterBand(1)
        slope_data = np.flipud(slope_band.ReadAsArray(col,row,width,height))
    else:
        slope_band = slope.GetRasterBand(1)
        slope_data = slope_band.ReadAsArray(col,row,width,height)
        
    
    VVmVH = VH_data*VV_data
    
    DEM_data = np.where(DEM_data >10000, 0, DEM_data)
    slope_data = np.where(slope_data >10000, 0, slope_data)

    merged = np.stack((VH_data, VV_data, VVmVH, NDWI_data, NDVI_data, DEM_data, slope_data), axis=2)
    # 关闭影像文件
    VH = None
    label = None
    VV = None
    DEM = None
    S2 = None
    slope = None
    return merged

## Register the VV and VH

def mergedImg(VH_path,VV_path,S2_path, DEM_path, Slope_path):
    # 打开VV,VH影像
    VH = gdal.Open(VH_path)
    VV = gdal.Open(VV_path)
    
    ##打开S2
    S2 = gdal.Open(S2_path)
    NDWI = S2.GetRasterBand(1)
    NDVI = S2.GetRasterBand(2)
    ##打开DEM
    DEM = gdal.Open(DEM_path)
    slope = gdal.Open(Slope_path)
    
    # 获取VH影像的范围
    Flip_label_VH = False
    VH_geo_transform = VH.GetGeoTransform()
    if VH_geo_transform[5]>0:
        VH_geo_transform = list(VH_geo_transform)
        VH_geo_transform[5] = -VH_geo_transform[5]
        Flip_label_VH = True
    
    # 获取S2影像的范围
    Flip_S2 = False
    S2_geo_transform = S2.GetGeoTransform()
    if S2_geo_transform[5]>0:
        Flip_S2 = True
        
    # 获取DEM影像的范围
    Flip_DEM = False
    DEM_geo_transform = DEM.GetGeoTransform()
    if DEM_geo_transform[5]>0:
        Flip_DEM = True
        
    # 获取DEM影像的范围
    Flip_slope = False
    DEM_geo_transform = DEM.GetGeoTransform()
    if DEM_geo_transform[5]>0:
        Flip_DEM = True
    
    # 获取slope影像的范围
    Flip_slope = False
    slope_geo_transform = slope.GetGeoTransform()
    if slope_geo_transform[5]>0:
        Flip_slope = True
    
    # 获取slope影像的范围
    Flip_label_VV = False
    VV_geo_transform = VV.GetGeoTransform()
    if VV_geo_transform[5]>0:
        Flip_label_VV = True
    
    
    # 读取相交区域的影像数据
    if Flip_label_VH:
        VH_data = np.flipud(VH.ReadAsArray())
    else:
        VH_data = VH.ReadAsArray()
    
    if Flip_label_VV:
        VV_data = np.flipud(VV.ReadAsArray())
    else:
        VV_data = VV.ReadAsArray()
    
    if Flip_S2:
        NDWI_data = np.flipud(NDWI.ReadAsArray())
        NDVI_data = np.flipud(NDVI.ReadAsArray())
    else:
        NDWI_data = NDWI.ReadAsArray()
        NDVI_data = NDVI.ReadAsArray()
    
    if Flip_DEM:
        DEM_band = DEM.GetRasterBand(1)
        DEM_data = np.flipud(DEM_band.ReadAsArray())
    else:
        DEM_band = DEM.GetRasterBand(1)
        DEM_data = DEM_band.ReadAsArray()
        
    if Flip_slope:
        slope_band = slope.GetRasterBand(1)
        slope_data = np.flipud(slope_band.ReadAsArray())
    else:
        slope_band = slope.GetRasterBand(1)
        slope_data = slope_band.ReadAsArray()
        
    
    VVmVH = VH_data*VV_data
    
    DEM_data = np.where(DEM_data >10000, 0, DEM_data)
    slope_data = np.where(slope_data >10000, 0, slope_data)

    merged = np.stack((VH_data, VV_data, VVmVH, NDWI_data, NDVI_data, DEM_data, slope_data), axis=2)
    # 关闭影像文件
    VH = None
    VV = None
    DEM = None
    S2 = None
    slope = None
    return merged




# Define img features
def imlabel_feature(labels,intensity_image):
    props = measure.regionprops(labels,intensity_image=intensity_image)
    object_ids = []
    im_area = []
    im_perimeter = []
    im_wid = []
    im_len = []
    im_ER = []
    im_P2A= []
    im_compactness = []
    im_SI = []
    im_LW = []
    im_eccentricity = []
    im_solidity = []
    im_rectangularity = []
    im_IPQ = []
    im_SR = []
    im_PFD = []
    im_SqP = []
    im_Asy = []
    im_BI = []
    im_VH_mean = []
    im_VV_mean = []
    im_VVmVH_mean = []
    im_NDWI_mean = []
    im_NDVI_mean = []
    im_Elevation_mean = []
    im_Slope_mean = []
    im_VH_std = []
    im_VV_std = []
    im_VVmVH_std = []
    im_NDWI_std = []
    im_NDVI_std = []
    im_Elevation_std = []
    im_Slope_std = []
    
    for region in props:
        #Area
        label_area = region.area
        im_area.append(label_area)
        #Perimeter
        label_perimeter = region.perimeter
        if label_perimeter == 0:
            label_perimeter = 1
            im_perimeter.append(label_perimeter)
        else:
            im_perimeter.append(label_perimeter)
        #ER
        im_ER.append(region.extent)
        #Label
        object_ids.append(region.label)
        #P2A
        label_P2A = label_perimeter*label_perimeter/(label_area)
        im_P2A.append(label_P2A)
        #SI
        label_SI = label_perimeter*label_perimeter/(4*np.sqrt(label_area))
        im_SI.append(label_SI)
        #Compactness
        label_Compactness = 2*np.sqrt(math.pi*label_area)/(label_perimeter)
        im_compactness.append(label_Compactness)
        #LW
        im_L = region.axis_major_length
        im_W = region.axis_minor_length
        if (im_L == 0) or (im_W == 0):
            LW = 0
        else:
            LW = np.double(im_L)/(np.double(im_W))
        im_wid.append(im_W)
        im_len.append(im_L)
        im_LW.append(LW)
        
        # eccentricity
        im_eccentricity.append(region.eccentricity)
        
        # Solidity
        im_solidity.append(region.solidity)
        
        # rectangularity
        rectangularity =label_area/(region.area_bbox)
        im_rectangularity.append(rectangularity)
        
        #IPQ (Iso-Perimetric Quotient)
        IPQ = 4*label_area*math.pi/label_perimeter**2
        im_IPQ.append(IPQ)
        
        # SR (Shape roughness)
        SR = label_perimeter/math.pi*(1+(im_L+im_W)/2)
        im_SR.append(SR)
        
        # PFD Patch Fractal Dimensions
        if label_area ==1:
            PFD = 0
        else:
            PFD = 2*math.log(label_perimeter/4)/math.log(label_area)
        im_PFD.append(PFD)
        
        #SqP (Square pixel metric)
        SqP = 1-4*np.sqrt(label_area)/label_P2A
        im_SqP.append(SqP)
        
        # Asy (Asymmetry)
        if im_L == 0:
            Asy = 0
        else:
            Asy = 1-im_W/im_L
        im_Asy.append(Asy)
        
        # BI (Boundary index)
        if im_L+im_W == 0:
            BI =0
        else:
            BI = label_perimeter/2*(im_L+im_W)
        im_BI.append(BI)
        
        # Intensity-image-mean
        mean_intensities = region.intensity_mean
        
        ## Sentinel-1
        VH_mean = mean_intensities[0]
        VV_mean = mean_intensities[1]
        VVmVH_mean = mean_intensities[2]
        ##Sentinel-2
        NDWI_mean = mean_intensities[3]
        NDVI_mean = mean_intensities[4]
        ##DEM
        Elevation_mean = mean_intensities[5]
        Slope_mean = mean_intensities[6]
        
        im_VH_mean.append(VH_mean)
        im_VV_mean.append(VV_mean)
        im_VVmVH_mean.append(VVmVH_mean)
        im_NDWI_mean.append(NDWI_mean)
        im_NDVI_mean.append(NDVI_mean)
        im_Elevation_mean.append(Elevation_mean)
        im_Slope_mean.append(Slope_mean)
        
        # Intensity-image-std
        std_intensities = region.intensity_std
        VH_std = std_intensities[0]
        VV_std = std_intensities[1]
        VVmVH_std = std_intensities[2]
        ##Sentinel-2
        NDWI_std = std_intensities[3]
        NDVI_std = std_intensities[4]
        ##DEM
        Elevation_std = std_intensities[5]
        Slope_std = std_intensities[6]
        
        im_VH_std.append(VH_std)
        im_VV_std.append(VV_std)
        im_VVmVH_std.append(VVmVH_std)
        im_NDWI_std.append(NDWI_std)
        im_NDVI_std.append(NDVI_std)
        im_Elevation_std.append(Elevation_std)
        im_Slope_std.append(Slope_std)
        
        
    im_feature = pd.DataFrame({'labels':object_ids, 'area':im_area, 'perimeter':im_perimeter, 'P2A':im_P2A, 'SI':im_SI, 'compactness':im_compactness, 'ER':im_ER,'LW':im_LW, 'width':im_wid,'length':im_len, 'eccentricity':im_eccentricity, 'Solidity':im_solidity, 'rectangularity':im_rectangularity, 'IPQ': im_IPQ, 'SR':im_SR, 'PFD':im_PFD, 'SqP':im_SqP, 'Asy':im_Asy, 'BI':im_BI,'VH_mean':im_VH_mean,'VV_mean':im_VV_mean, 'VVmVH_mean':im_VVmVH_mean, 'NDWI_mean':im_NDWI_mean, 'NDVI_mean':im_NDVI_mean, 'Elevation_mean':im_Elevation_mean, 'Slope_mean':im_Slope_mean,'VH_std':im_VH_std,'VV_std':im_VV_std, 'VVmVH_std':im_VVmVH_std, 'NDWI_std':im_NDWI_std, 'NDVI_std':im_NDVI_std, 'Elevation_std':im_Elevation_std, 'Slope_std':im_Slope_std}, columns = ['labels', 'area', 'perimeter', 'P2A', 'SI', 'compactness', 'ER','LW','width','length','eccentricity','Solidity','rectangularity','IPQ','SR','PFD','SqP','Asy','BI','VH_mean','VV_mean','VVmVH_mean', 'NDWI_mean', 'NDVI_mean', 'Elevation_mean', 'Slope_mean', 'VH_std', 'VV_std', 'VVmVH_std', 'NDWI_std', 'NDVI_std', 'Elevation_std', 'Slope_std'], index = object_ids)
        
    return im_feature






if __name__ == '__main__':
    
    
    VH_filepath = ''
    VV_filepath = ''
    S2_filepath = ''
    dem_filepath = ''
    slope_filepath = ''
    SR_filepath = ''
    result_name = ''
    ndwi_filepath = ''
    
    VH_data, VH_width, VH_height, VH_bands, VH_geotransform, VH_projection = ReadTif(VH_filepath)
    VV_data, VV_width, VV_height, VV_bands, VV_geotransform, VV_projection = ReadTif(VV_filepath)
    
    SR_data, SR_width, SR_height, SR_bands, SR_geotransform, SR_projection = ReadTif(SR_filepath)
    ndwi_data, ndwi_width, ndwi_height, ndwi_bands, ndwi_geotransform, ndwi_projection = ReadTif(ndwi_filepath)
    
    VH_data = np.nan_to_num(VH_data, nan=np.nanmean(VH_data))
    VV_data = np.nan_to_num(VV_data, nan=np.nanmean(VV_data))
    
    VVmVH = VH_data*VV_data
    
    ##Load model
    rfc = joblib.load(r'random_forest_model-v1-py3.pkl')
    
    
    img_nor = normalized(VVmVH, VH_height, VH_width)
    water_extent = waterextract(img_nor)
    
    SR = SR_data>0.4
    nowater = ndwi_data<-0.1
    condition = SR | nowater
    water_extent[condition] = 0
    
    water = morphology.remove_small_holes(water_extent,100)
    labels = measure.label(water, background = 0, connectivity = 1)
    Filter_labels = morphology.remove_small_objects(labels, min_size = 10, connectivity= 2)
    merged = mergedImg(VH_filepath,VV_filepath,S2_filepath,dem_filepath,slope_filepath)
    label_feature = imlabel_feature(Filter_labels,merged)
    input_feature = label_feature.iloc[:,1:]
    input_feature = input_feature.dropna(subset=['VV_mean','VH_mean'],how='all')
    ##Predict
    predict = rfc.predict(input_feature)
    predict1 = predict.tolist()
    label_Id=0
    pre_result = Filter_labels.copy()
    object_ids = label_feature['labels'].to_list()
    for predict_result in predict1:
        if predict_result ==1:
            Id = object_ids[label_Id]
            pre_result[pre_result==Id] = 1
        else:
            Id = object_ids[label_Id]
            pre_result[pre_result==Id] = 0
        label_Id+=1
    Savetiff(pre_result, VH_width, VH_height, VH_bands, VH_geotransform, VH_projection, result_name)

    
    
    
      

        
    
    
    
    
