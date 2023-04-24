# [1] - Liu, Yongxue, et al. "Detecting high-temperature anomalies from Sentinel-2 MSI images." ISPRS Journal of Photogrammetry and Remote Sensing 177 (2021): 174-193.
# The next functions implement a simplified version of the algorithm described in [1].

import torch
import torchvision

def get_TAI(s2_l1c_img):
    """Returns the th Thermal Anomaly Index (TAI) as defined in [1].


    Args:
        s2_l1c_img (torch.tensor): Sentinel2-L1C image. 

    Returns:
        torch.tensor: TAI (with negative values saturated a 0)
    """

    tai=torch.div(s2_l1c_img[:,:,2] - s2_l1c_img[:,:,1],  s2_l1c_img[:,:,0])

    return torch.where(tai >= torch.zeros_like(tai), tai, torch.zeros_like(tai))

def get_TAI_mean(tai):
    """It returns the TAI mean as defined in [1]. TAI_mean is calculated by using a 15x15 convolutional filter. The input image is padded with the values at the boundaries to avoid boundaries false positives.

    Args:
        tai (torch.tensor): tai index

    Returns:
        torch.tensor: convoluted alert-map
    """
    conv=torch.nn.Conv2d(1, 1, 31)
    weight=torch.nn.Parameter(1/31 * torch.ones([31,31]).unsqueeze(0).unsqueeze(0),requires_grad=False)
    
    if tai.device.type=='cuda':
        conv=conv.cuda()
    conv.load_state_dict({'weight': weight, 'bias' : torch.zeros([1])}, strict=False)
    with torch.no_grad():
        padding=torchvision.transforms.Pad((15,15,15,15), padding_mode='edge')
        img_pad=padding(tai)
        tai_mean=conv(torch.tensor(img_pad, dtype=torch.float32).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        del weight
        del conv
        del img_pad
        torch.cuda.empty_cache()
    return tai_mean


def get_TAI_buffer(tai, tai_mean):
    """Return the TAI buffer as in [1] for each active pixel.

    Args:
        tai (torch.tensor): TAI index.
        tai_mean (torch.tensor): TAI mean (over 15 pixel ray) buffer.

    Returns:
        torch.tensor: TAI buffer.
    """
    with torch.no_grad():
        s_index=torch.where(tai -tai_mean >= 0.45, 1, 0).nonzero()
        padding=torchvision.transforms.Pad((15,15,15,15), padding_mode='constant', fill=0)
        s_pad=padding(tai)
        b=torch.zeros_like(s_pad)
        for (r,c) in s_index:
            b[r-15:r+15,c-15:c+15]=1

        b=b[15:-15, 15:-15] #Removing edge pixels having no information.

        return b


def cluster_9px(img):
    """It performs the convolution to detect clusters of 9 activate pixels (current pixel and 8 surrounding pixels) are at 1. 

    Args:
        img (torch.tensor): input alert-matrix

    Returns:
        torch.tensor: convoluted alert-map
    """
    
    conv=torch.nn.Conv2d(1, 1, 3)
    weight=torch.nn.Parameter(torch.tensor([[[[1.0,1.0,1.0], [1.0,1.0,1.0], [1.0,1.0,1.0]]]]),requires_grad=False)
    img_pad=torch.nn.functional.pad(img, (1,1,1,1), mode='constant', value=1)
    if img.device.type=='cuda':
        conv=conv.cuda()
    conv.load_state_dict({'weight': weight, 'bias' : torch.zeros([1])}, strict=False)
    with torch.no_grad():
        surrounded=conv(torch.tensor(img_pad, dtype=torch.float32).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        #Adding clipping to the mean
        surrounded=torch.where(surrounded >= 0.45, 0.45 * torch.ones_like(surrounded), surrounded)
        del weight
        del conv
        del img_pad
        torch.cuda.empty_cache()
    return surrounded

def manage_saturated_thermal_anomalies(s2_l1c_img):
    """It checks thermal anomalies as described in [1] by returning a mask to control not-8-connected thermal anomalies.

    Args:
        s2_l1c_img (torch.tensor): Sentinel2 L1C image.

    Returns:
        torch.tensor: saturated thermal anomalies mask
    """
    saturated_mask=torch.logical_and(torch.where(s2_l1c_img[:,:,2] >= 1, 1, 0), torch.where(s2_l1c_img[:,:,2] >= 1,1,0))
    saturated_mask_8_connected=cluster_9px(saturated_mask)
    return torch.logical_or(saturated_mask_8_connected, torch.logical_not(saturated_mask))

def get_image_lost_source_mask(s2_l1c_img):
    """It removes pixels with lost packages as in [1] by returning a mask.

    Args:
        s2_l1c_img (torch.tensor): Sentinel2 L1C image.

    Returns:
        torch.tensor: nask to exclude image lost.
    """
    source_lost_mask_negated=torch.logical_or(torch.where(s2_l1c_img[:,:,0] < 0.01, 1, 0), torch.where(s2_l1c_img[:,:,1] < 0.05,1,0))
    return torch.logical_not(source_lost_mask_negated)

def remove_false_positives(s2_l1c_img, thermal_anomalies_map):
    """Removing false positives due to saturated surfaces, image packet loss or edge anomalies.

    Args:
        s2_l1c_img (torch.tensor): Sentinel2 L1C image
        thermal_anomalies_map (torch.tensor): thermal anomalies binary map to clean

    Returns:
        torch.tensor: clean thermal_anomalies_map
    """
    thermal_saturated_mask=manage_saturated_thermal_anomalies(s2_l1c_img)
    thermal_saturated_mask_no_saturation=torch.logical_and(thermal_anomalies_map, thermal_saturated_mask)
    source_lost_mask=get_image_lost_source_mask(s2_l1c_img)
    thermal_anomalies_no_lost_no_saturation=torch.logical_and(thermal_saturated_mask_no_saturation, source_lost_mask)
    #Removing pixel affected by NaN
    nan_mask=torch.logical_not(torch.logical_or(torch.logical_or(torch.isnan(s2_l1c_img[:,:,0]),torch.isnan(s2_l1c_img[:,:,1])),torch.isnan(s2_l1c_img[:,:,2])))
    thermal_anomalies_clean_map=torch.logical_and(thermal_anomalies_no_lost_no_saturation, nan_mask)
    return thermal_anomalies_clean_map


def extract_high_tempearature_anomalies(s2_l1c_img):
    """Extracts the hight temperature anomaly map as in [1] without time-series check.

    Args:
        s2_l1c_img (torch.tensor): Sentinel2 L1C image.

    Returns:
        torch.tensor: high-temperature anomalis map
    """
    tai=get_TAI(s2_l1c_img)
    tai_mean=get_TAI_mean(tai)
    b=get_TAI_buffer(tai, tai_mean)
    thr_1=torch.logical_and(torch.where(tai>= 0.45 * torch.ones_like(tai), torch.ones_like(tai), torch.zeros_like(tai)), b) #  Selecting pixels in the buffer meeting TAI >= 0.45
    thr_2=torch.where(s2_l1c_img[:,:,2] > 2 * s2_l1c_img[:,:,1]  - s2_l1c_img[:,:,0], torch.ones_like(tai) ,torch.zeros_like(tai))  # Condition p_far_SWIR - p_near_SWIR > p_near_SWIR - p_NIR.
    thr_3=torch.where(s2_l1c_img[:,:,2] > 0.15 * torch.ones_like(tai), torch.ones_like(tai), torch.zeros_like(tai)) #  Removing low reflectivity surfaces in FAR SWIR
    thermal_anomaly_map=torch.logical_and(torch.logical_and(thr_1, thr_2), thr_3)
    thermal_anomaly_clean_map=remove_false_positives(s2_l1c_img, thermal_anomaly_map)
    return thermal_anomaly_clean_map, thermal_anomaly_map