# -*- coding:utf-8 -*-
"""
@author:shifuxiao
@code_version: 2.0.0
"""
import argparse
import numpy as np
from scipy.ndimage import label
from scipy import ndimage
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
import torch
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.cuda.is_available())
import torchio as tio
from skimage import measure, morphology
from sklearn.cluster import KMeans
import math
from lungtumormask import mask as ltmask
from medpy.metric.binary import hd
from nibabel import load
import cv2
import lxml.etree as ET
from lungmask import LMInferer
# import nibabel as nib
# from totalsegmentator.python_api import totalsegmentator



class Breath:
    def __init__(self):
        self.createTime = time.localtime()
        self.condication = {}
        self.lung_split = "Double_lung"

    def resample(self,imgs, spacing, new_spacing, mode="linear"):
        """
        :return: new_image, true_spacing
        """
        dim = len(imgs.shape)
        if dim == 3 or dim == 2:
            # If the image is 3D or 2D image
            # Use torchio.Resample to resample the image.

            # Create a sitk Image object then load this object to torchio Image object
            imgs_itk = sitk.GetImageFromArray(imgs)
            # print("输出spacing:",np.flipud(spacing))
            imgs_itk.SetSpacing(np.flipud(spacing).astype(np.float64))
            # imgs_itk.SetSpacing(spacing.astype(np.float64))
            imgs_tio = tio.ScalarImage.from_sitk(imgs_itk)

            # Resample Image
            print("输出new_spacing:", list(np.flipud(new_spacing)))

            resampler = tio.Resample(list(np.flipud(new_spacing)), image_interpolation=mode)
            new_imgs = resampler(imgs_tio).as_sitk()

            # Prepare return value
            new_spacing = new_imgs.GetSpacing()
            print("new_spacing",new_spacing)
            new_imgs = sitk.GetArrayFromImage(new_imgs)
            resize_factor = np.array(imgs.shape) / np.array(new_imgs.shape)
            return new_imgs, new_spacing, resize_factor
        elif dim == 4:
            # If the input is a batched 3D image
            # Run resample on each image in the batch.
            n = imgs.shape[-1]
            newimg = []
            for i in range(n):
                slice = imgs[:, :, :, i]
                newslice, true_spacing, resize_factor = self.resample(slice, spacing, new_spacing, mode=mode)
                newimg.append(newslice)
            newimg = np.transpose(np.array(newimg), [1, 2, 3, 0])
            print("输出newimg的shape:", newimg.shape)
            return newimg, true_spacing, resize_factor
        else:
            raise ValueError('wrong shape')

    def seg_bg_mask(self,img):
        """
        Calculate the segementation mask for the whole body.
        Assume the dimensions are in Superior/inferior, anterior/posterior, right/left order.
        :param img: a 3D image represented in a numpy array.
        :return: The segmentation Mask. BG = 0
        """
        (D, W, H) = img.shape

        img_cp = np.copy(img)
        mean = np.mean(img)
        std = np.std(img)
        img = img - mean
        img = img / std
        # Find the average pixel value near the lungs
        # to renormalize washed out images
        middle = img[int(D / 5):int(D / 5 * 4), int(W / 5):int(W / 5 * 4), int(H / 5):int(H / 5 * 4)]
        mean = np.mean(middle)
        # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image
        # clear bg
        dilation = morphology.dilation(thresh_img, np.ones([4, 4, 4]))
        eroded = morphology.erosion(dilation, np.ones([4, 4, 4]))
        # Select the largest area besides the background
        labels = measure.label(eroded, background=1)
        regions = measure.regionprops(labels)
        roi_label = 0
        max_area = 0
        for region in regions:
            if region.label != 0 and region.area > max_area:
                max_area = region.area
                roi_label = region.label
        thresh_img = np.where(labels == roi_label, 1, 0)

        # bound the ROI.
        # TODO: maybe should check for bounding box
        # thresh_img = 1 - eroded
        sum_over_traverse_plane = np.sum(thresh_img, axis=(1, 2))
        top_idx = 0
        for i in range(D):
            if sum_over_traverse_plane[i] > 0:
                top_idx = i
                break
        bottom_idx = D - 1
        for i in range(D - 1, -1, -1):
            if sum_over_traverse_plane[i] > 0:
                bottom_idx = i
                break
        for i in range(top_idx, bottom_idx + 1):
            thresh_img[i] = morphology.convex_hull_image(thresh_img[i])
        labels = measure.label(thresh_img)
        bg_labels = []
        corners = [(0, 0, 0), (-1, 0, 0), (0, -1, 0), (-1, -1, 0), (0, -1, -1), (0, 0, -1), (-1, 0, -1), (-1, -1, -1)]
        for pos in corners:
            bg_labels.append(labels[pos])
        bg_labels = np.unique(np.array(bg_labels))

        mask = labels
        for l in bg_labels:
            mask = np.where(mask == l, -1, mask)
        mask = np.where(mask == -1, 0, 1)

        roi_labels = measure.label(mask, background=0)
        roi_regions = measure.regionprops(roi_labels)
        bbox = [0, 0, 0, D, W, H]
        for region in roi_regions:
            if region.label == 1:
                bbox = region.bbox

        return mask, bbox

    def seg_lung_mask(self,img):
        """
        Calculate the segementation mask either for lung only.
        :param img: a 3D image represented in a numpy array.
        :return: The segmentation Mask.
        """
        (D, W, H) = img.shape

        mean = np.mean(img)
        std = np.std(img)
        img = img - mean
        img = img / std
        # Find the average pixel value near the lungs
        # to renormalize washed out images
        middle = img[int(D / 5):int(D / 5 * 4), int(W / 5):int(W / 5 * 4), int(H / 5):int(H / 5 * 4)]
        mean = np.mean(middle)
        img_max = np.max(img)
        img_min = np.min(img)
        # To improve threshold finding, I'm moving the
        # underflow and overflow on the pixel spectrum
        img[img == img_max] = mean
        img[img == img_min] = mean
        # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image
        # # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.
        # # We don't want to accidentally clip the lung.
        eroded = morphology.erosion(thresh_img, np.ones([4, 4, 4]))
        dilation = morphology.dilation(eroded, np.ones([4, 4, 4]))
        labels = measure.label(dilation)
        label_vals = np.unique(labels)
        regions = measure.regionprops(labels)
        good_regions = []
        for prop in regions:
            B = prop.bbox
            if (B[4] - B[1] < W / 20 * 18 and B[4] - B[1] > W / 5 and B[4] < W / 20 * 16 and B[1] > W / 10 and
                    B[5] - B[2] < H / 20 * 18 and B[5] - B[2] > H / 20 and B[2] > H / 10 and B[5] < H / 20 * 19 and
                    B[3] - B[0] > D / 4):
                good_regions.append(prop)
                continue
                print(B)

            if (B[4] - B[1] < W / 20 * 18 and B[4] - B[1] > W / 6 and B[4] < W / 20 * 18 and B[1] > W / 20 and
                    B[5] - B[2] < H / 20 * 18 and B[5] - B[2] > H / 20):
                good_regions.append(prop)
                continue
            if B[4] - B[1] < W / 20 * 18 and B[4] - B[1] > W / 20 and B[4] < W / 20 * 18 and B[1] > W / 20:
                good_regions.append(prop)
                continue
        # Select the most greatest region
        good_regions = sorted(good_regions, key=lambda x: x.area, reverse=True)

        mask = np.ndarray([D, W, H], dtype=np.int8)
        mask[:] = 0
        #  After just the lungs are left, we do another large dilation
        #  in order to fill in and out the lung mask
        good_labels_bbox = []
        for N in good_regions[:2]:
            mask = mask + np.where(labels == N.label, 1, 0)
            good_labels_bbox.append(N.bbox)

        # Get the bbox of lung
        bbox = [D / 2, W / 2, H / 2, D / 2, W / 2, H / 2]
        for b in good_labels_bbox:
            for i in range(0, 3):
                bbox[i] = min(bbox[i], b[i])
                bbox[i + 3] = max(bbox[i + 3], b[i + 3])

        mask = morphology.dilation(mask, np.ones([4, 4, 4]))  # one last dilation
        mask = morphology.erosion(mask, np.ones([4, 4, 4]))

        return mask, bbox

    def normalize_intensity(self,img, linear_clip=False, clip_range=None):
        """
        a numpy image, normalize into intensity [0,1]
        (img-img.min())/(img.max() - img.min())
        :param img: image
        :param linear_clip:  Linearly normalized image intensities so that the 95-th percentile gets mapped to 0.95; 0 stays 0
        :return:
        """

        if linear_clip:
            if clip_range is not None:
                img[img < clip_range[0]] = clip_range[0]
                img[img > clip_range[1]] = clip_range[1]
                normalized_img = (img - clip_range[0]) / (clip_range[1] - clip_range[0])
            else:
                img = img - img.min()
                normalized_img = img / np.percentile(img, 95) * 0.95
        else:
            # If we normalize in HU range of softtissue
            min_intensity = img.min()
            max_intensity = img.max()
            normalized_img = (img - img.min()) / (max_intensity - min_intensity)
        return normalized_img

    def resize_image_itk(self, ori_img, target_img, resamplemethod=sitk.sitkNearestNeighbor):
        """
        用itk方法将原始图像resample到与目标图像一致
        :param ori_img: 原始需要对齐的itk图像
        :param target_img: 要对齐的目标itk图像
        :param resamplemethod: itk插值方法: sitk.sitkLinear-线性  sitk.sitkNearestNeighbor-最近邻
        :return:img_res_itk: 重采样好的itk图像
        """
        target_Size = target_img.GetSize()  # 目标图像大小  [x,y,z]
        target_Spacing = target_img.GetSpacing()  # 目标的体素块尺寸    [x,y,z]
        target_origin = target_img.GetOrigin()  # 目标的起点 [x,y,z]
        target_direction = target_img.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]

        # itk的方法进行resample
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ori_img)  # 需要重新采样的目标图像
        # 设置目标图像的信息
        resampler.SetSize(target_Size)  # 目标图像大小
        resampler.SetOutputOrigin(target_origin)
        resampler.SetOutputDirection(target_direction)
        resampler.SetOutputSpacing(target_Spacing)
        # 根据需要重采样图像的情况设置不同的dype
        if resamplemethod == sitk.sitkNearestNeighbor:
            resampler.SetOutputPixelType(sitk.sitkUInt8)  # 近邻插值用于mask的，保存uint8
        else:
            resampler.SetOutputPixelType(sitk.sitkFloat32)  # 线性插值用于PET/CT/MRI之类的，保存float32
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(resamplemethod)
        itk_img_resampled = resampler.Execute(ori_img)  # 得到重新采样后的图像
        return itk_img_resampled

    def interpolate(self, volumeImage, newSpacing):
        '''
        插值处理
        :para volumeImage:volume格式
        :para newSpaceing:想要插值或者重采样处理后得到的物理间隔，格式为列表
        :return newVolumeImage:插值处理后的到的volume文件
        '''
        resampleFilter = sitk.ResampleImageFilter()
        resampleFilter.SetInterpolator(sitk.sitkLinear)  ##此处为线性插值，其他插值方式可以去官网查询
        resampleFilter.SetOutputDirection(volumeImage.GetDirection())
        resampleFilter.SetOutputOrigin(volumeImage.GetOrigin())
        newSpacing = np.array(newSpacing, float)
        newSize = volumeImage.GetSize() / newSpacing * volumeImage.GetSpacing()
        newSize = newSize.astype(np.int)
        resampleFilter.SetSize(newSize.tolist())
        resampleFilter.SetOutputSpacing(newSpacing)
        newVolumeImage = resampleFilter.Execute(volumeImage)
        return newVolumeImage

    def normalize_intensity(self, img, linear_clip=False):
        # TODO: Lin-this line is for CT. Modify it to make this method more general.
        img[img < -1024] = -1024
        return img

    def keep_largest_tumor_region(self, mask_path,dataset_name):
        """
        Given a 3D binary mask representing a lung tumor, this function retains only
        the largest connected tumor region.

        Parameters:
        - mask (np.ndarray): A 3D numpy array where non-zero values represent
                             the tumor regions.

        Returns:
        - largest_region (np.ndarray): A 3D numpy array with the largest
                                       connected tumor region.
        """
        mask = sitk.ReadImage(mask_path)
        spacing = mask.GetSpacing()
        origin = mask.GetOrigin()
        direction = mask.GetDirection()
        mask = sitk.GetArrayFromImage(mask)
        # Label the connected regions in the 3D mask
        labeled_mask, num_features = label(mask)
        print("num_tumor:{}#".format(num_features))
        child_nodes = {"tumor_number":num_features}
        self.generate_xml_file("{}_tumor_segmentation _output".format(dataset_name),child_nodes)
        # self.write_xml("{}_tumor_segmentation _output".format(dataset_name),num_features)
        # If there are no features, return an empty mask
        if num_features == 0:
            return np.zeros_like(mask)
        # Find the index/label of the largest tumor region
        largest_region_label = 1
        max_volume = 0
        for region_label in range(1, num_features + 1):
            # Compute the size of the region with current label
            region_volume = np.sum(labeled_mask == region_label)
            # Update the label of the largest region found so far
            if region_volume > max_volume:
                max_volume = region_volume
                largest_region_label = region_label
        # Generate the mask for the largest region
        largest_region = (labeled_mask == largest_region_label).astype(mask.dtype)
        center_of_mass = ndimage.measurements.center_of_mass(largest_region)
        # print("center_of_mass",center_of_mass)
        largest_region = sitk.GetImageFromArray(largest_region)
        largest_region.SetSpacing(spacing)
        largest_region.SetOrigin(origin)
        largest_region.SetDirection(direction)
        mask_save = os.path.splitext(mask_path)[0] + "_max.nii.gz"
        sitk.WriteImage(largest_region, mask_save)
        return largest_region

    def caculate_tumor_center(self,input_path,dataset_name):
        mask = sitk.ReadImage(input_path)
        mask = sitk.GetArrayFromImage(mask)
        center_of_mass = ndimage.center_of_mass(mask)
        self.condication["tumor_center"] = center_of_mass
        child_nodes = {"tumor_center":center_of_mass}
        self.generate_xml_file("{}_caculate_tumor_center_output".format(dataset_name),child_nodes)
        # self.write_xml("tumor_center",center_of_mass)
        # print("tumor_center:",center_of_mass)
        # return center_of_mass


    def multiply(self, source, source_seg, mask_save):
        # scan = sitk.ReadImage(scan_dir)
        scan_array = sitk.GetArrayFromImage(source)
        # print("source_seg",source_seg )
        mask_array = sitk.GetArrayFromImage(source_seg)
        lungSave = np.multiply(scan_array, mask_array)
        scan_array[mask_array == 0] = 0
        lungSave = sitk.GetImageFromArray(lungSave)
        lungSave.SetSpacing(source.GetSpacing())
        lungSave.SetOrigin(source.GetOrigin())
        lungSave.SetDirection(source.GetDirection())
        # print("lungSave",lungSave)
        # print("mask_save",mask_save)
        if mask_save != None:
            sitk.WriteImage(lungSave, mask_save)
        return lungSave

    def read_cbct_data_list(self, data_folder_path):
        case_list = os.listdir(data_folder_path)
        return_list = []
        for case in case_list:
            # case_dir = os.path.join(data_folder_path, case + '/' + case)
            case_data = os.path.join(data_folder_path, case)
            return_list.append(case_data)
        return return_list

    def process_single_file(self,output, path_pair, sz, spacing, seg_bg=False):
        # print("输出path_pair:", path_pair)
        parts = path_pair.split("\\")
        # print(parts)
        last_part = parts[-1]
        # 使用字符串分割方法，按点号分割文件名，获取不包括扩展名的部分
        file_name_parts = last_part.split(".")
        desired_part = file_name_parts[0]
        # if not os.path.exists(output):
        #     os.makedirs(output)
        new_path = os.path.join(output, desired_part + "_KMeans" + ".nii.gz")
        new_pathSeg = os.path.join(output,
                                   desired_part + "_Seg" + ".nii.gz")
        source_origin_img = sitk.ReadImage(path_pair)
        data = sitk.GetArrayFromImage(source_origin_img)
        ori_spacing = np.array(source_origin_img.GetSpacing())[::-1]
        ori_sz = np.array(source_origin_img.GetSize())[::-1]
        ori_dir = source_origin_img.GetDirection()
        print("输出data的shape:", data.shape)
        print("输出ori_spacing:", ori_spacing)
        print("输出ori_sz:", ori_sz)
        print("输出ori_dir:", ori_dir)
        source = data
        
        target = data
        source, _, _ = self.resample(source, ori_spacing, spacing)
        print("source shape:",source.shape)
        source[source < -1024] = -1024
        target[target < -1024] = -1024
        bbox = [0, 0, 0] + list(ori_sz)
        if seg_bg:
            (D, W, H) = ori_sz
            bg_hu = np.min(source)
            source_bg_seg, source_bbox = self.seg_bg_mask(source)
            source[source_bg_seg == 0] = bg_hu
            total_voxel = np.prod(target.shape)
            print("##########Area percentage of ROI:{:.2f}".format(
                float(np.sum(source_bg_seg)) / total_voxel))
        source_seg, _ = self.seg_lung_mask(source)


        # Pad 0 if shape is smaller than desired size.
        new_origin = np.array((0, 0, 0))
        sz_diff = sz - source.shape
        sz_diff[sz_diff < 0] = 0
        pad_width = [[int(sz_diff[0] / 2), sz_diff[0] - int(sz_diff[0] / 2)],
                     [int(sz_diff[1] / 2), sz_diff[1] - int(sz_diff[1] / 2)],
                     [int(sz_diff[2] / 2), sz_diff[2] - int(sz_diff[2] / 2)]]
        source = np.pad(source, pad_width, constant_values=-1024)
        source_seg = np.pad(source_seg, pad_width, constant_values=0)
        new_origin[sz_diff > 0] = -np.array(pad_width)[sz_diff > 0, 0]
        # Crop if shape is greater than desired size.
        sz_diff = source.shape - sz
        bbox = [[int(sz_diff[0] / 2), int(sz_diff[0] / 2) + sz[0]],
                [int(sz_diff[1] / 2), int(sz_diff[1] / 2) + sz[1]],
                [int(sz_diff[2] / 2), int(sz_diff[2] / 2) + sz[2]]]
        source = source[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
        source_seg = source_seg[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
        new_origin[sz_diff > 0] = np.array(bbox)[sz_diff > 0, 0]
        source = self.normalize_intensity(source)
        file_name = os.path.basename(path_pair)
        ###############
        index = file_name.find(".nii.gz")
        print("输出index:", index)
        print("输出source的shape:", source.shape)
        # 颠倒坐标轴（x, y, z）
        source = sitk.GetImageFromArray(source)
        source_seg = sitk.GetImageFromArray(source_seg)
        #fill_hole
        dilate_filter = sitk.BinaryDilateImageFilter()
        # 设置膨胀滤波器的参数
        dilate_filter.SetKernelRadius(12)  # 设置卷积核半径,根据孔洞大小灵活设置
        dilate_filter.SetForegroundValue(1)  # 设置前景值为1
        dilate_filter.SetBackgroundValue(0)  # 设置背景值为0
        # 应用膨胀滤波器
        dilated_image = dilate_filter.Execute(source_seg)
        # 创建腐蚀滤波器
        erode_filter = sitk.BinaryErodeImageFilter()
        # 设置腐蚀滤波器的参数
        erode_filter.SetKernelRadius(12)  # 设置卷积核半径
        erode_filter.SetForegroundValue(1)  # 设置前景值为1
        erode_filter.SetBackgroundValue(0)  # 设置背景值为0
        # 应用腐蚀滤波器
        source_seg = erode_filter.Execute(dilated_image)
        print("输出spacing:", spacing[::-1])
        # source.SetSpacing(spacing[::-1])
        # source_seg.SetSpacing(spacing[::-1])
        source.SetSpacing([1.25, 1.25, 1.25])
        source_seg.SetSpacing([1.25, 1.25, 1.25])
        # source.SetDirection(ori_dir)
        # source.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
        print("sourceDirection",source.GetDirection())
        # source_seg.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
        sitk.WriteImage(source, new_path)
        sitk.WriteImage(source_seg, new_pathSeg)
        return source, target, source_seg, _, new_origin, _

    def preprocess(self, input, output,case_num=200):
        data_folder_path = input
        # if not os.path.exists(data_folder_path):
        #     print("Did not find data list file at %s" % data_folder_path)
        #     return
        # file_list = self.read_cbct_data_list(data_folder_path)
        # data_count = len(file_list)
        # print("输出file_list:", file_list)
        # for i in range(data_count):
        #     if "after" in file_list[i] or "before" in file_list[i] or "middle" in file_list[i]:

        # if "after" in file_list[i] or "before" in file_list[i] or "middle" in file_list[i]:
            # print("输出file_list[i]：", file_list[i])
        file_name = os.path.basename(input)
        file_name = os.path.splitext(file_name)[0]
        # print("file_name", file_name)
        try:
            source, target, source_seg, target_seg, new_origin, new_spacing = self.process_single_file(
                                                                                                       output,
                                                                                                       input,
                                                                                                       np.array((
                                                                                                           256, 256,
                                                                                                           256)),
                                                                                                       np.array(
                                                                                                           (1.25, 1.25,
                                                                                                            1.25)),
                                                                                                       seg_bg=True,
                                                                                                       )

            mask_save = os.path.join(os.path.abspath(output_path), file_name + "lungRegion.nii.gz")
            # mask_fill = os.path.join(os.path.abspath(output_path), file_name + "lungRegion.nii.gz")
            # source_seg = self.fill_hole(source_seg,mask_fill)
            self.multiply(source, source_seg, mask_save)
        except ValueError as err:
            print(err, "input image error")

    # return source, target, source_seg, target_seg, new_origin, new_spacing

    def add_tumor_mask(self, mask_array, tumor_mask_file):
        tumor_mask_Image = sitk.ReadImage(tumor_mask_file)
        tumor_mask_array = sitk.GetArrayFromImage(tumor_mask_Image)

        all_array = mask_array + tumor_mask_array
        mask_image = sitk.GetImageFromArray(all_array)
        mask_image.SetOrigin(tumor_mask_Image.GetOrigin())
        mask_image.SetSpacing(tumor_mask_Image.GetSpacing())
        mask_image.SetDirection(tumor_mask_Image.GetDirection())
        all_mask_path = os.path.splitext(tumor_mask_file)[0] + "_tumor_add_needle.nii.gz"
        # print("all_mask_path", tumor_mask_file)
        sitk.WriteImage(mask_image, all_mask_path)
        return mask_image

    def threshold_seg(self, data_path):

        # 获取图像的像素数据
        data_nii = sitk.ReadImage(data_path)
        srcArray = sitk.GetArrayFromImage(data_nii)
        threshold = 2000
        # 创建一个新的SimpleITK图像
        srcImage = sitk.GetImageFromArray(np.squeeze(srcArray > threshold).astype(np.int16))
        # 设置图像的原点、间距和方向，如果有的话
        srcImage.SetOrigin(data_nii.GetOrigin())
        srcImage.SetSpacing(data_nii.GetSpacing())
        print("data_nii.GetSpacing()", data_nii.GetSpacing())
        return srcImage

    def morphSwitch(self, srcImage, input_file,dataset_name):
        # 连通性阈值
        # Initialize variables to keep track of the farthest points
        pointsConnect = []
        labelConnectArray = []
        thrConnect_1 = 1000
        thrConnect_2 = 30000
        # 执行膨胀操作
        morphRadius = 1
        structDilate = sitk.BinaryDilateImageFilter()
        structDilate.SetKernelRadius(morphRadius)
        dilateImage = structDilate.Execute(srcImage)
        structErode = sitk.BinaryErodeImageFilter()
        structErode.SetKernelRadius(morphRadius)
        dilateImage = structErode.Execute(dilateImage)
        morphImage = dilateImage
        # 创建一个ConnectedComponentImageFilter实例
        connectedComponentFilter = sitk.ConnectedComponentImageFilter()
        # 对erodedImage应用连通性分析
        connectedImage = connectedComponentFilter.Execute(morphImage)
        # 获取连通组件的数量
        label_image = sitk.LabelShapeStatisticsImageFilter()
        label_image.Execute(connectedImage)
        # print("label_image.GetLabels:",label_image.GetLabels())
        for label in label_image.GetLabels():
            centroid = label_image.GetCentroid(label)
            print(f"Object {label} Centroid: {centroid}")

            # Get the mask for the current label
            mask = sitk.BinaryThreshold(connectedImage, label, label)
            # Convert the mask to a NumPy array
            mask_array = sitk.GetArrayFromImage(mask)
            # Find the coordinates of all the pixels in the mask
            coordinates = np.argwhere(mask_array == 1)
            print(coordinates.shape)
            coordinates[:, [0, 1, 2]] = coordinates[:, [2, 1, 0]]
            # 保存为文本文件
            labelConnectArray.append(coordinates)
            max_distance = 0
            farthest_point1 = None
            farthest_point2 = None
            # Calculate the distance for each point and find the two farthest points in each direction
            print("coordinates:",len(coordinates))
            execute_second_logic = True
            if thrConnect_1 != 0 or thrConnect_2 != 0:
                if len(coordinates) > thrConnect_2 or len(coordinates) < thrConnect_1:
                    execute_second_logic = False #检测是否有针
                else:
                    execute_second_logic = True
            if (execute_second_logic):
                self.save_mask(mask_array, srcImage, input_file)  # 保存消融针mask
                # print("save_mask over")
                for point1 in coordinates:
                    x1, y1, z1 = point1
                    for point2 in coordinates:
                        x2, y2, z2 = point2
                        # Calculate the distances in X, Y, and Z directions
                        distance_x = abs(x2 - x1)
                        distance_y = abs(y2 - y1)
                        distance_z = abs(z2 - z1)
                        # Calculate the 3D Euclidean distance between the two points
                        distance_3d = np.sqrt(distance_x ** 2 + distance_y ** 2 + distance_z ** 2)
                        if distance_3d > max_distance:
                            max_distance = distance_3d
                            farthest_point1 = point1
                            farthest_point2 = point2
                pointsConnect.append([farthest_point1, farthest_point2])
                child_nodes = {"needle":pointsConnect}
                self.generate_xml_file("{}_needle_segmentation_output".format(dataset_name),child_nodes)
                print("pointsConnect:",pointsConnect)
                # At this point, farthest_point1 and farthest_point2 will be the two points with the maximum 3D distance
                print("The two farthest points in 3D space:")
                print("Point 1:", farthest_point1)
                print("Point 2:", farthest_point2)
                # try:
                #     self.add_tumor_mask(mask_array, tumor_mask_file)  # 加上
                # except RuntimeError as err:
                #     print(err,"please input tumor mask")
        print("pointsConnect length:", len(pointsConnect))
        print("pointsConnect:", pointsConnect)
        return pointsConnect

    def save_mask(self, mask_array, srcimage, input_file):
        mask_image = sitk.GetImageFromArray(mask_array)
        mask_image.SetOrigin(srcimage.GetOrigin())
        mask_image.SetSpacing(srcimage.GetSpacing())
        mask_image.SetDirection(srcimage.GetDirection())
        needle_mask_path = os.path.splitext(input_file)[0] + "needle.nii.gz"
        # print("needle_mask_path", needle_mask_path)
        sitk.WriteImage(mask_image, needle_mask_path)
        # return mask_image
    # def total_segmentation(self,input_path,output_path,lung_split = "all"):
    #     if lung_split == "all":
    #         roi_subset_lung = ["lung_upper_lobe_left","lung_upper_lobe_right","lung_lower_lobe_left","lung_lower_lobe_right","lung_middle_lobe_right"]
    #     elif lung_split == "right":
    #         roi_subset_lung = ["lung_upper_lobe_right", "lung_lower_lobe_right", "lung_middle_lobe_right"]
    #     elif lung_split == "left":
    #         roi_subset_lung = ["lung_upper_lobe_left","lung_lower_lobe_left"]
    #     # totalsegmentator(input_path, output_path, ml=True,device='gpu',roi_subset=roi_subset_lung)
    #     # output_img = totalsegmentator(input_path, output_path, ml=True,roi_subset=roi_subset_lung)
    #     output_img = totalsegmentator(input_path, output_path, ml=True,roi_subset=roi_subset_lung)
    #     # output_img[output_img >0 ] = 1
    #     return output_img
    def CBCT_lungseg_nnUNet(self,input_path, output_path, tumor_path,lung_segPath):
        parts = input_path.split("\\")
        last_part = parts[-1]
        # 使用字符串分割方法，按点号分割文件名，获取不包括扩展名的部分
        file_name_parts = last_part.split(".")
        file_name = file_name_parts[0]
        input_image = sitk.ReadImage(input_path)
        #做resample
        self.crop_volume(lung_segPath,lung_segPath,file_name)
        lung_mask = sitk.ReadImage(lung_segPath)
        spacing = input_image.GetSpacing()
        mask_save = os.path.join(os.path.abspath(output_path), file_name + ".niilungRegion.nii.gz")
        lung_Splite = self.isSingleOrDouble_lung(lung_mask)
        child_nodes = {"lung_splite":lung_Splite}
        self.generate_xml_file("lung_split_output",child_nodes)
        if lung_Splite =="Single_lung":
            self.seg_single_lung(lung_mask, tumor_path, input_image, mask_save, spacing,file_name,output_path)
        else:
            self.seg_double_lung(lung_mask, tumor_path, input_image, mask_save, spacing, file_name, output_path)
        # self.seg_single_lung(lung_mask, tumor_path, input_image, mask_save, spacing, file_name, output_path)

    def lung_segmentation(self, input_path, output_path,modality,tumor_path,lung_segPath):

        if modality == "CT":
            self.CT_lungseg_R231(input_path,output_path,tumor_path)

        elif modality == "CBCT":

            self.CBCT_lungseg_nnUNet(input_path, output_path, tumor_path,lung_segPath)



    def tumor_segmentation(self, input_file,output_file,dataset_name):

        raw_resam_file = input_file.replace("input_data","output")
        resp_out_path = "\\".join(raw_resam_file.split("\\")[0:-1])
        file_name= raw_resam_file.split("\\")[-1]
        print(file_name)
        if not os.path.exists(resp_out_path):
            os.makedirs(resp_out_path)
        print("raw_resam_path:",resp_out_path)
        self.crop_volume(input_file, raw_resam_file,file_name)
        try:
            ltmask.mask(input_file, output_file, "--lung-filter", 0.3, 3, 8)
            self.crop_volume(output_file, output_file,file_name)
            # largest_region = self.keep_largest_tumor_region(output_file,dataset_name)
        except OverflowError as err:
            print(err,"plese change image")
        # return largest_region


    def tumor_save(self,input_file, mask_tumor,output_file):

        input_image = sitk.ReadImage(input_file)
        tumor_file = sitk.ReadImage(mask_tumor)
        # print(output_file)
        self.multiply(input_image,tumor_file,output_file)
        print("tumor save over")


    def needle_segmentation(self, input_file,dataset_name):
        srcImage = self.threshold_seg(input_file)
        pointsConnect = self.morphSwitch(srcImage, input_file,dataset_name)
        self.condication["pointsConnect"] = pointsConnect
        return pointsConnect

    def rigid_registration(self, fixed_image_path, moving_image_path, moving_image_mask, moving_image_tumor_mask_file,dataset_name,surgeryStatus):
        # # 读取两个CT图像
        fixed_image = sitk.ReadImage(fixed_image_path,
                                     sitk.sitkFloat32)
        moving_image = sitk.ReadImage(moving_image_path,
                                      sitk.sitkFloat32)
        moving_image_mask = sitk.ReadImage(moving_image_mask,
                                           sitk.sitkFloat32)
        moving_image_tumor_mask = sitk.ReadImage(moving_image_tumor_mask_file,sitk.sitkFloat32)  # zl
        spacing = fixed_image.GetSpacing()
        direction = fixed_image.GetDirection()
        origin = fixed_image.GetOrigin()

        def start_plot():
            global metric_values, multires_iterations

            metric_values = []
            multires_iterations = []

        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

        moving_resampled = sitk.Resample(
            moving_image,
            fixed_image,
            initial_transform,
            sitk.sitkLinear,
            0.0,
            moving_image.GetPixelID(),
        )

        # # Callback invoked when the EndEvent happens, do cleanup of data and figure.
        def end_plot():
            global metric_values, multires_iterations

            del metric_values
            del multires_iterations
            # Close figure, we don't want to get a duplicate of the plot latter on.
            # plt.close()

        def plot_values(registration_method):
            global metric_values, multires_iterations
            metric_values.append(registration_method.GetMetricValue())
            # Clear the output area (wait=True, to reduce flickering), and plot current data

        def update_multires_iterations():
            global metric_values, multires_iterations
            multires_iterations.append(len(metric_values))

        def display_images_with_alpha(image_z, alpha, fixed, moving):
            img = (1.0 - alpha) * fixed[:, :, image_z] + alpha * moving[:, :, image_z]
        #     plt.imshow(sitk.GetArrayViewFromImage(img), cmap=plt.cm.Greys_r)
        #     plt.axis("off")
        #     plt.show()

        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            sitk.Euler3DTransform(),
            # sitk.CenteredTransformInitializerFilter.GEOMETRY, #几何中心
            sitk.CenteredTransformInitializerFilter.MOMENTS,#质心
        )

        registration_method = sitk.ImageRegistrationMethod()

        # Similarity metric settings.
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50) #越大精度高。直方图精细程度，计算互信息
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.2)

        registration_method.SetInterpolator(sitk.sitkLinear)

        # Optimizer settings.
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=2,
            numberOfIterations=200,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Setup for the multi-resolution framework.
        # registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        # registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Don't optimize in-place, we would possibly like to run this cell multiple times.
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        # Connect all of the observers so that we can perform plotting during registration.
        registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
        registration_method.AddCommand(
            sitk.sitkMultiResolutionIterationEvent, update_multires_iterations
        )
        registration_method.AddCommand(
            sitk.sitkIterationEvent, lambda: plot_values(registration_method)
        )

        final_transform = registration_method.Execute(
            sitk.Cast(fixed_image, sitk.sitkFloat32), sitk.Cast(moving_image, sitk.sitkFloat32)
        )
        print(f"Final metric value: {registration_method.GetMetricValue()}")
        print(
            f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}"
        )
        parts = fixed_image_path.split("\\")
        OUTPUT_DIR = "\\".join(parts[0:-1])
        print("OUTPUT_DIR:", OUTPUT_DIR)
        # transformed_moving = sitk.TransformGeometry(moving_image, final_transform)
        transformed_moving = sitk.Resample(moving_image, moving_image.GetSize(), final_transform, sitk.sitkLinear,origin, spacing)
        # moving_image_mask = sitk.TransformGeometry(moving_image_mask, final_transform)
        # moving_image_tumor_mask = sitk.TransformGeometry(moving_image_tumor_mask, final_transform)
        moving_image_mask = sitk.Resample(moving_image_mask, moving_image_mask.GetSize(), final_transform,
                                          sitk.sitkNearestNeighbor, origin, spacing)
        moving_image_tumor_mask = sitk.Resample(moving_image_tumor_mask, moving_image_tumor_mask.GetSize(),
                                                final_transform, sitk.sitkNearestNeighbor, origin, spacing)
        # moving_image_tumor_mask = sitk.Resample(moving_image_tumor_mask, moving_image_tumor_mask.GetSize(),
        #                                         final_transform, sitk.sitkLinear, origin, spacing)
        # moving_image_tumor_mask[moving_image_tumor_mask > 0] =1
        # transformed_moving = sitk.TransformGeometry(moving_image, final_transform)
        print(
            f"origin before: {moving_image.GetOrigin()}\norigin after: {transformed_moving.GetOrigin()}"
        )
        print(
            f"direction cosine before: {moving_image.GetDirection()}\ndirection cosine after: {transformed_moving.GetDirection()}"
        )
        sitk.WriteImage(
            transformed_moving,
            os.path.join(OUTPUT_DIR, dataset_name + r"_moved.nii.gz"),
        )
        ##########################
        # transformed_moving.SetSpacing(spacing)
        # transformed_moving.SetOrigin(origin)
        # transformed_moving.SetDirection(direction)
        moving_image_tumor_mask.SetSpacing(spacing)
        # moving_image_tumor_mask.SetOrigin(origin)
        # print("origin",origin)
        # moving_image_tumor_mask.SetDirection(direction) #zl
        moving_image_mask.SetSpacing(spacing)
        # moving_image_mask.SetOrigin(origin)
        # moving_image_mask.SetDirection(direction)

        sitk.WriteImage(
            moving_image_mask,
            os.path.join(OUTPUT_DIR, dataset_name + r"_movedMaskRagid_{}.nii.gz".format(surgeryStatus)),
        )

        sitk.WriteImage(
            moving_image_tumor_mask,
            os.path.join(OUTPUT_DIR, dataset_name + r"_movedTumorMaskRagid_{}.nii.gz".format(surgeryStatus)),  # zl
        )
        mask_image = moving_image_mask-moving_image_tumor_mask
        sitk.WriteImage(
            mask_image,
            os.path.join(OUTPUT_DIR, dataset_name + r"_movedSubMaskRagid_{}.nii.gz".format(surgeryStatus)),
        )
        return transformed_moving,moving_image_mask,moving_image_tumor_mask

    def remove_tumor(self,regidImage,regidTumorMask,surgeryStatus,dataset_name):
        parts = regidImage.split("\\")
        mask_save = os.path.join("\\".join(parts[0:-1]), dataset_name + "mask_subTumor" + surgeryStatus + ".nii.gz")
        scan = sitk.ReadImage(regidImage)
        imageDirection = scan.GetDirection()
        spacing = scan.GetSpacing()
        scanArray = sitk.GetArrayFromImage(scan)
        print("输出scanArray的shape:", scanArray.shape)
        mask = sitk.ReadImage(regidTumorMask)
        # print("mask", mask.GetDirection())
        # print("mask_ori", mask.GetOrigin())
        maskArray = sitk.GetArrayFromImage(mask)
        maskArray[maskArray > 0] = 1
        scan_cp = np.copy(scanArray)
        scan_cp[maskArray > 0] = 0
        # scanArray = scanArray - scanArray * maskArray
        # lungSave = sitk.GetImageFromArray(scanArray.astype("float32"))
        lungSave = sitk.GetImageFromArray(scan_cp.astype("float32"))
        lungSave.SetSpacing(mask.GetSpacing())
        lungSave.SetDirection(mask.GetDirection())
        lungSave.SetOrigin(mask.GetOrigin())
        sitk.WriteImage(lungSave, mask_save)
        return lungSave

    def norigid_registration(self, fixed_image, moving_image, fixed_mask, moving_mask, moved_mask, dataset_name):
        parts = fixed_image.split("\\")
        OUTPUT_DIR = "\\".join(parts[0:-1])
        # print("OUTPUT_DIR:", OUTPUT_DIR)
        fixed_image = sitk.ReadImage(fixed_image, sitk.sitkFloat32)
        moving_image = sitk.ReadImage(moving_image,sitk.sitkFloat32)
        moved_tumor_mask = sitk.ReadImage(moving_mask,sitk.sitkFloat32)
        after_tumor_mask = sitk.ReadImage(fixed_mask,sitk.sitkFloat32)
        fixed_mask = sitk.ReadImage(fixed_mask,sitk.sitkFloat32)
        moved_mask = sitk.ReadImage(moved_mask,sitk.sitkFloat32)
        moving_mask = sitk.ReadImage(moving_mask,sitk.sitkFloat32)
        # moving_maskArray = sitk.GetArrayFromImage(moving_image)
        # fixed_maskArray = sitk.GetArrayFromImage(fixed_image)
        matcher = sitk.HistogramMatchingImageFilter()
        if fixed_image.GetPixelID() in (sitk.sitkUInt8, sitk.sitkInt8):
            matcher.SetNumberOfHistogramLevels(128)
        else:
            matcher.SetNumberOfHistogramLevels(1024)
        matcher.SetNumberOfMatchPoints(7)#精度有关，匹配点数据量相关
        # matcher.SetNumberOfMatchPoints(15)
        matcher.ThresholdAtMeanIntensityOn()
        moving_matched = matcher.Execute(moving_image, fixed_image)

         # Fast symmetric forces Demons Registration
        demons = sitk.FastSymmetricForcesDemonsRegistrationFilter() #加速的demos
        demons.SetNumberOfIterations(200) #时间约束
         # 设置位移场的高斯平滑标准差
        demons.SetStandardDeviations(2.0) #可变形位移场的非刚性配准，
        # demons.SetStandardDeviations(0.5) #可变形位移场的非刚性配准，

        # demons.SetFixedImageMask(fixed_mask)
        # demons.SetMovingImageMask(moving_mask)
            # 执行配准
        displacement_field = demons.Execute(fixed_image, moving_matched) #位移场
        # sitk.WriteImage(displacement_field, os.path.join(OUTPUT_DIR,dataset_name + 'displacement_field.nii.gz'))
        final_transform = sitk.DisplacementFieldTransform(displacement_field)
        # 5. 可选：重采样移动图像
        resampled_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                        moving_image.GetPixelID())

        # 6. 保存结果（可选）
        sitk.WriteImage(resampled_image, os.path.join(OUTPUT_DIR,dataset_name + "_movedNoRigidFFD.nii.gz"))

        moved_mask = sitk.Resample(
            moved_mask,
            fixed_mask,
            final_transform,
            sitk.sitkNearestNeighbor,
            0.0,
            moving_image.GetPixelID(),
        )

        moved_tumor_mask = sitk.Resample(
            moved_tumor_mask,
            after_tumor_mask,
            final_transform,
            sitk.sitkNearestNeighbor,
            0.0,
            moving_image.GetPixelID(),
        )
        moved_tumor_mask[moved_tumor_mask > 0] = 1
        spacing = resampled_image.GetSpacing()

        moved_mask.SetSpacing(spacing)
        # moved_mask = sitk.TransformGeometry(moved_mask, final_transform)
        sitk.WriteImage(
            moved_mask, os.path.join(OUTPUT_DIR, dataset_name + "_movedTumorMaskNorigidFFD.nii.gz")
        )

        moved_tumor_mask.SetSpacing(spacing)
        # moved_tumor_mask = sitk.TransformGeometry(moved_tumor_mask, final_transform)
        sitk.WriteImage(
            moved_tumor_mask,
            os.path.join(OUTPUT_DIR, dataset_name + "_movedMaskNorigidFFD.nii.gz"),
        )
        return resampled_image,moved_mask,moved_tumor_mask


    def difference_image(self,scan_dir,norigidTumor_mask,moved,dataset_name):
        parts = scan_dir.split("\\")
        OUTPUT_DIR = "\\".join(parts[0:-1])
        print("OUTPUT_DIR:", OUTPUT_DIR)
        scan = sitk.ReadImage(scan_dir)
        spacing = scan.GetSpacing()
        origin = scan.GetOrigin()
        direction = scan.GetDirection()
        scanArray = sitk.GetArrayFromImage(scan)
        print("输出scanArray的shape:", scanArray.shape)
        moved = sitk.ReadImage(moved)
        spacing = scan.GetSpacing()
        moved = sitk.GetArrayFromImage(moved)
        mask = sitk.ReadImage(norigidTumor_mask)
        maskArray = sitk.GetArrayFromImage(mask)
        maskArray[maskArray > 0] = 1
        scan_cp = np.copy(scanArray)
        moved[maskArray > 0] == 0
        scan_cp[maskArray > 0] == 0
        difference = moved - scan_cp
        scanArray[maskArray == 0] == 0
        # save =difference[maskArray>0]
        difference[maskArray > 0] = scanArray[maskArray > 0]
        difference = sitk.GetImageFromArray(difference)
        difference.SetSpacing(spacing)
        difference.SetOrigin(origin)
        difference.SetDirection(direction)
        save_path = os.path.join(OUTPUT_DIR,r"{}_difference.nii.gz".format(dataset_name))
        sitk.WriteImage(difference, save_path)
        return difference
    def difference_add(self,nii_file_path,maskFile,dataset_name):
        parts = maskFile.split("\\")
        OUTPUT_DIR = "\\".join(parts[0:-1])
        print("OUTPUT_DIR:", OUTPUT_DIR)
        nii_file_path = sitk.ReadImage(nii_file_path, sitk.sitkFloat32)
        spacing = nii_file_path.GetSpacing()
        origin = nii_file_path.GetOrigin()
        direction = nii_file_path.GetDirection()

        input_image_path = sitk.GetArrayFromImage(nii_file_path)
        image_array = np.array(input_image_path)
        ##################################

        maskFile = sitk.ReadImage(maskFile, sitk.sitkFloat32)
        maskFile = sitk.GetArrayFromImage(maskFile)
        # maskFile[maskFile > 0] = 1
        image_array[maskFile > 0] = maskFile[maskFile > 0]
        ##################################
        gray_image = image_array
        gray_image = sitk.GetImageFromArray(gray_image)
        gray_image.SetOrigin(origin)
        gray_image.SetDirection(direction)
        gray_image.SetSpacing(spacing)
        save_path = os.path.join(OUTPUT_DIR,dataset_name+ "difference_addMask.nii.gz")
        print("save_path",save_path)
        sitk.WriteImage(gray_image,save_path)
        return gray_image

    def caculate_distance(self, point1, point2):
        print("point1[0]",point1[0])
        distance = math.sqrt((float(point2[0]) - float(point1[0])) ** 2 + (float(point2[1]) - float(point1[1])) ** 2 + (float(point2[2]) - float(point1[2])) ** 2)
        return distance

    def remove_mixedTag(self,mask_image,save_path=None):
        # image = sitk.ReadImage(mask_path)
        # 创建腐蚀滤波器
        erode_filter = sitk.BinaryErodeImageFilter()
        # 设置腐蚀滤波器的参数
        erode_filter.SetKernelRadius(6)  # 设置卷积核半径
        erode_filter.SetForegroundValue(1)  # 设置前景值为1
        erode_filter.SetBackgroundValue(0)  # 设置背景值为0
        # 应用腐蚀滤波器
        eroded_image = erode_filter.Execute(mask_image)
        dilate_filter = sitk.BinaryDilateImageFilter()
        # 设置膨胀滤波器的参数
        dilate_filter.SetKernelRadius(6)  # 设置卷积核半径,根据孔洞大小灵活设置
        dilate_filter.SetForegroundValue(1)  # 设置前景值为1
        dilate_filter.SetBackgroundValue(0)  # 设置背景值为0
        # 应用膨胀滤波器
        dilated_image = dilate_filter.Execute(eroded_image)
        # 保存结果图像
        if save_path != None:
            sitk.WriteImage(dilated_image, save_path)
        return dilated_image

    def fill_hole(self,mask_path):
        # image = sitk.ReadImage(mask_path)
        # 创建膨胀滤波器
        dilate_filter = sitk.BinaryDilateImageFilter()
        # 设置膨胀滤波器的参数
        dilate_filter.SetKernelRadius(8)  # 设置卷积核半径,根据孔洞大小灵活设置
        dilate_filter.SetForegroundValue(1)  # 设置前景值为1
        dilate_filter.SetBackgroundValue(0)  # 设置背景值为0
        # 应用膨胀滤波器
        dilated_image = dilate_filter.Execute(mask_path)
        # 创建腐蚀滤波器
        erode_filter = sitk.BinaryErodeImageFilter()
        # 设置腐蚀滤波器的参数
        erode_filter.SetKernelRadius(8)  # 设置卷积核半径
        erode_filter.SetForegroundValue(1)  # 设置前景值为1
        erode_filter.SetBackgroundValue(0)  # 设置背景值为0
        # 应用腐蚀滤波器
        eroded_image = erode_filter.Execute(dilated_image)
        # 保存结果图像
        # sitk.WriteImage(eroded_image, save_path)
        return eroded_image

    # 找出含有肿瘤轮廓上的坐标
    def zllkzb(self,mask):  # 用来返回肿瘤轮廓上的坐标
        Serial_Number = []  # 初始化一个列表，用来存储含有肿瘤的切片序号（z值）
        Total_Number = 0  # 初始化一个变量，用来计数含有肿瘤的切片的个数，最后据此取含有肿瘤的中间切片进行计算
        for i in range(mask.shape[2]):  # 找出喊肿瘤的一组切片
            if mask[:, :, i].any() == True:
                Serial_Number.append(i)
                Total_Number += 1
        zong_con = []
        # print(mask.shape)#(256, 256, 256)
        for num in range(Total_Number):
            mask_part = mask[:, :, num + Serial_Number[0]]
            mask_8u = mask_part.astype(np.uint8)
            contours, _ = cv2.findContours(mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = np.array(contours)  # 将tuple类型转成numpy类型
            # print('切片序号：',num+Serial_Number[0])
            # print('提取轮廓：',contours.shape)
            # print(type(contours))#<class 'numpy.ndarray'>
            if contours.shape[0] == 1:  # 此时的contours类似于（1，坐标个数，1,2），两个1都是冗余的，最后的2一个是x一个是y
                contour = contours[0, :, 0, :]  # 直接去除冗余的维度
                complete_con = [list(row) + [num + Serial_Number[0]] for row in contour]  # 堆叠z值
                # print(type(complete_con))
                # print(type(zong_con))
                zong_con += complete_con
                # zong_con.append(complete_con)#错误方法，会形成嵌套列表
            elif contours.shape[0] > 1:
                contours = np.vstack(contours)  # 将子数组堆叠，堆叠结果类似于（坐标个数，1,2），中间的1是冗余的，最后的2一个是x一个是y
                contour = contours[:, 0, :]
                complete_con = [list(row) + [num + Serial_Number[0]] for row in contour]  # 堆叠z值
                zong_con += complete_con
                # zong_con.append(complete_con)#错误方法，会形成嵌套列表
            elif contours.shape[0] == 0:  # mask并非完全的二值图像，这时是含有肿瘤，但肿瘤边界不明显的，师兄说忽略
                contour = []  # 不计入
            zong_con = np.array(zong_con)
            zong_con = zong_con.tolist()
        return zong_con

    def distance(self,point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)

    def find_closest_points(self,points1, points2):
        min_distance = float('inf')
        closest_pair = None
        for point1 in points1:
            for point2 in points2:
                dist = self.distance(point1, point2)
                if dist < min_distance:
                    min_distance = dist
                    closest_pair = (point1, point2)
        return min_distance, closest_pair

    def Hausdorff_3d(self,mask1, mask2):
        # 计算豪斯多夫距离
        hausdorff_distance = hd(mask1, mask2)
        return hausdorff_distance

    def dis_bor(self,mask1, mask2,dataset_name):
        mask1 = sitk.ReadImage(mask1)
        mask2 = sitk.ReadImage(mask2)
        mask1 = self.set_origin_to_zero(mask1)
        mask2 = self.set_origin_to_zero(mask2)
        mask1 = sitk.GetArrayFromImage(mask1).astype('u1')
        mask2 = sitk.GetArrayFromImage(mask2).astype('u1')
        mask1[mask1 > 0] = 1
        mask2[mask2 > 0] = 1
        # 寻找最小距离及对应的点对
        ZB1 = self.zllkzb(mask1)  # mask1的肿瘤边界坐标list
        ZB2 = self.zllkzb(mask2)  # mask2的肿瘤边界坐标list
        min_dist, closest_pair = self.find_closest_points(ZB1, ZB2)
        hausdorff_distance = self.Hausdorff_3d(mask1, mask2)
        # self.write_xml('{} distance_hd _output',hausdorff_distance * 1.25)
        # self.write_xml(" distance_hd _output",min_dist * 1.25)
        child_nodes = {"Dmax": str(hausdorff_distance * 1.25), "Dmin": str(min_dist * 1.25)}
        self.generate_xml_file("{}_distance_hd_output".format(dataset_name),child_nodes)
        # print('Dmax：', hausdorff_distance, 'px;', hausdorff_distance * 1.25, 'mm#')
        # print('Dmin：', min_dist, 'px;', min_dist * 1.25, 'mm#')
        return min_dist, hausdorff_distance

    def load_nifti(self,file_path):
        image = sitk.ReadImage(file_path)
        data = sitk.GetArrayFromImage(image)
        return image, data

    def abnormal_erase(self,mask,raw_image,save_path,dataset_name,surgeryStatus):

        mask = sitk.ReadImage(mask)
        raw_image = sitk.ReadImage(raw_image)
        origin = raw_image.GetOrigin()
        spacing = raw_image.GetSpacing()
        direction = raw_image.GetDirection()
        raw_data = self.multiply(mask,raw_image,mask_save=None)
        raw_array = sitk.GetArrayFromImage(raw_image)
        image_array = sitk.GetArrayFromImage(raw_data)
        raw = raw_array - image_array
        image = sitk.GetImageFromArray(raw)
        image.SetOrigin(origin)
        image.SetSpacing(spacing)
        image.SetDirection(direction)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path,"{}_{}.niilungRegion.nii.gz".format(dataset_name,surgeryStatus))
        sitk.WriteImage(image, save_path)
        return image

    def set_origin_to_zero(self,image):
        current_origin = image.GetOrigin()
        spacing = image.GetSpacing()
        new_origin = (0.0, 0.0, 0.0)
        translation_vector = [new_origin[i] - current_origin[i] for i in range(3)]
        translation_transform = sitk.TranslationTransform(3, translation_vector)
        image = sitk.Resample(image, image.GetSize(), translation_transform, sitk.sitkLinear, current_origin, spacing)
        image.SetOrigin(new_origin)
        return image

    def cov_ratio(self,tumor_mask_file,norigidTumor_mask,dataset_name):   
        image_mid = sitk.ReadImage(tumor_mask_file)
        image_before = sitk.ReadImage(norigidTumor_mask)
        # image_mid_origin = self.set_origin_to_zero(image_mid)
        # image_before_origin = self.set_origin_to_zero(image_before)
        data_mid = sitk.GetArrayFromImage(image_mid).astype('u1')
        data_before = sitk.GetArrayFromImage(image_before).astype('u1')
        intersect_V = np.sum(data_mid * data_before)
        before_V = np.sum(data_before)
        cover_ratio = (intersect_V / before_V)*100
        volume_ratio = np.sum(data_mid) / before_V
        child_nodes = {"cover_ratio": cover_ratio, "volume_ratio": volume_ratio}
        self.generate_xml_file("{}_cov_ratio_output".format(dataset_name),child_nodes)
        print("coverage_ratio:%.2f" % cover_ratio + "#")
        print("volume_ratio:%.2f" % volume_ratio + "#")
        return cover_ratio,volume_ratio
    def statistical_value(self,mask1):
        mask1 = sitk.ReadImage(mask1)
        mask_array = sitk.GetArrayFromImage(mask1)
        mask_array[mask_array > 0] = 1
        value = np.sum(mask_array) * 1.25 * 1.25 * 1.25
        return value
    def generate_xml_file(self,name,child_nodes):
        root = ET.Element("root")
        for node_name, node_text in child_nodes.items():
            node = ET.SubElement(root, node_name)
            node.text = str(node_text)

        tree = ET.ElementTree(root)
        file_name = f"{name}.xml"
        tree.write(file_name, pretty_print=True, xml_declaration=True, encoding="utf-8")
        print(f"XML file '{file_name}' has been generated.")

    def set_origin(self,input_path,out_path):
        image = sitk.ReadImage(input_path)
        image_ori = self.set_origin_to_zero(image)
        sitk.WriteImage(image_ori, out_path)
    def lung_mask(self,mask1,save_path,surgeryStatus,dataset_name):
        mask1 = sitk.ReadImage(mask1)
        origin = mask1.GetOrigin()
        direction = mask1.GetDirection()
        spacing = mask1.GetSpacing()
        mask_array = sitk.GetArrayFromImage(mask1)
        mask_array[mask_array > 0] = 1
        image = sitk.GetImageFromArray(mask_array)
        image.SetOrigin(origin)
        image.SetSpacing(spacing)
        image.SetDirection(direction)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, "{}_{}_seg.nii.gz".format(dataset_name, surgeryStatus))
        sitk.WriteImage(image, save_path)

    def itkResampleBySpacing(self,itkimage, newSpacing, resamplemethod=sitk.sitkLinear):
        resampler = sitk.ResampleImageFilter()
        originSpacing = itkimage.GetSpacing()
        print("Spacing", originSpacing)
        originSize = itkimage.GetSize()
        print("originSize", originSize)
        ratio = originSpacing / newSpacing
        newSize = np.round(originSize * ratio)
        newSize = newSize.astype('int32')
        resampler.SetReferenceImage(itkimage)
        resampler.SetSize(newSize.tolist())
        resampler.SetOutputSpacing(newSpacing.tolist())
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(resamplemethod)
        itkimgResampled = resampler.Execute(itkimage)
        return itkimgResampled, ratio

    def itkPadding(self,itkimage, padding_size, value=0):
        data = sitk.GetArrayFromImage(itkimage)
        shape = np.array(data.shape)
        # assert (shape <= padding_size).all(), "Padding size should bigger than the image shape"
        padding = [0] * 3
        padding_left = [0] * 3
        padding_size = padding_size[::-1]
        for i in range(3):
            padding[i] = max(0, padding_size[i] - shape[i])
            padding_left[i] = max(0, padding[i] // 2)
        data_padding = np.pad(data, ((padding_left[0], padding[0] - padding_left[0]),
                                     (padding_left[1], padding[1] - padding_left[1]),
                                     (padding_left[2], padding[2] - padding_left[2])),
                              mode='constant',
                              constant_values=value
                              )
        itkimage_padding = sitk.GetImageFromArray(data_padding)
        itkimage_padding.SetSpacing(itkimage.GetSpacing())
        return itkimage_padding, padding_left[::-1]
    def itkCrop(self,itkimage, patch_size, residual):

        residual = residual[::-1]  # [W, H, D]
        itkimage_padding, padding_left = self.itkPadding(itkimage, patch_size)
        dim = np.array(itkimage_padding.GetSize())
        center = dim // 2 + np.array(residual).astype('int')
        bottom_legal = np.array(patch_size) // 2 - dim // 2
        top_legal = dim - np.array(patch_size) // 2 - dim // 2
        print('input_residual: [%.1f, %.1f, %.1f]' % (residual[2], residual[1], residual[0]))
        print('legal_residual_range: [%d ~ %d, %d ~ %d, %d ~ %d]' % (bottom_legal[2], top_legal[2],
                                                                     bottom_legal[1], top_legal[1],
                                                                     bottom_legal[0], top_legal[0],
                                                                     ))
        bottom = center - np.array(patch_size) // 2
        x0, y0, z0 = bottom
        x0 = min(max(x0, 0), dim[0] - patch_size[0])
        y0 = min(max(y0, 0), dim[1] - patch_size[1])
        z0 = min(max(z0, 0), dim[2] - patch_size[2])
        crop_left = [x0, y0, z0]
        return itkimage_padding[x0:x0 + patch_size[0], y0:y0 + patch_size[1],
               z0:z0 + patch_size[2]], crop_left, padding_left

    def threhold_mask(self,mask1, save_path):
        mask1 = sitk.ReadImage(mask1)
        origin = mask1.GetOrigin()
        direction = mask1.GetDirection()
        spacing = mask1.GetSpacing()
        mask_array = sitk.GetArrayFromImage(mask1)
        mask_array[mask_array > 0] = 1
        image = sitk.GetImageFromArray(mask_array)
        image.SetOrigin(origin)
        image.SetSpacing(spacing)
        image.SetDirection(direction)
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # save_path = os.path.join(save_path, "{}_{}_seg.nii.gz".format(dataset_name, surgeryStatus))
        sitk.WriteImage(image, save_path)

    def normalize(self,image, vmin=-1024, vmax=-3072):
        image = (image - vmin) / (vmax - vmin)
        image[image < 0] = 0
        image[image > 1] = 1
        return image

    def crop_volume(self,input_path,out_path=None,file_name=None):

        if os.path.isfile(input_path):
            image_std = sitk.ReadImage(input_path)
            # direction = image_std.GetDirection()
            dst_scale = [1.25, 1.25, 1.25]
            image_std, ratio = self.itkResampleBySpacing(image_std, np.array(dst_scale), resamplemethod=sitk.sitkLinear)
            dst_dim = [256, 256, 256]
            residual = [0, 0, 0]
            # 4. crop
            image_crop, crop_left, padding_left = self.itkCrop(image_std, dst_dim, residual)
            image_crop = sitk.Cast(image_crop, sitk.sitkFloat32)
            image_crop = sitk.GetArrayFromImage(image_crop)
            if ("after" in file_name) or ("middle" in file_name):
                image_crop = image_crop[::-1]
                image_crop = image_crop[:, :, ::-1]
                image_crop = sitk.GetImageFromArray(image_crop)
                image_crop.SetSpacing([1.25, 1.25, 1.25])
                # image_crop.SetDirection(direction)
                if out_path!=None:
                    sitk.WriteImage(image_crop, out_path)
                else:
                    pass
                    return image_crop
            else:
                image_crop = sitk.GetImageFromArray(image_crop)
                image_crop.SetSpacing([1.25, 1.25, 1.25])
                # image_crop.SetDirection(direction)
                if out_path != None:
                    sitk.WriteImage(image_crop, out_path)
                else:
                    pass
                    return image_crop


    def flip_mask_nii(self,mask_nii, save_nii):

        data = sitk.ReadImage(mask_nii)
        Sapcing = data.GetSpacing()
        nii = sitk.GetArrayFromImage(data)
        nii = np.flip(nii, 0)  # z
        nii = np.flip(nii, 2)
        result_img = sitk.GetImageFromArray(nii)
        # direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
        # result_img.SetDirection(direction)
        result_img.SetSpacing(Sapcing)
        sitk.WriteImage(result_img, save_nii)

    def copy_geometry(self,image,ref):
        image.SetOrigin(ref.GetOrigin())
        image.SetDirection(ref.GetDirection())
        image.SetSpacing(ref.GetSpacing())
        return image

    def CT_lungseg_R231(self,input_path,output_path,tumor_path):

        parts = input_path.split("\\")
        last_part = parts[-1]
        # 使用字符串分割方法，按点号分割文件名，获取不包括扩展名的部分
        file_name_parts = last_part.split(".")
        file_name = file_name_parts[0]
        input_image = sitk.ReadImage(input_path)
        spacing = input_image.GetSpacing()
        inferer = LMInferer()
        segmentation = inferer.apply(input_image)  # default model is U-net(R231)
        # 将numpy.ndarray类型的数据转换为sitk.Image类型
        img_out_itk = sitk.GetImageFromArray(segmentation)
        img_out_itk = self.copy_geometry(img_out_itk, input_image)
        # source_seg = img_out_itk[img_out_itk>0]=1
        # 保存处理后的图像
        # processed_img_path = os.path.join(output_folder,file_name + "_seg.nii.gz")
        # sitk.WriteImage(img_out_itk, processed_img_path)
        mask_save = os.path.join(os.path.abspath(output_path), file_name + ".niilungRegion.nii.gz")
        xml_file = "./lung_split_output.xml"
        lung_splite = self.read_xml_value(xml_file,value="lung_splite")
        print(lung_splite)
        if lung_splite =="Single_lung":
            self.seg_single_lung(img_out_itk, tumor_path, input_image, mask_save, spacing,file_name,output_path)
        else:
            self.seg_double_lung(img_out_itk, tumor_path, input_image, mask_save, spacing, file_name, output_path)

    def isSingleOrDouble_lung(self,lung_image):

        left_lung_count = np.sum(sitk.GetArrayFromImage(lung_image) == 2)
        right_lung_count = np.sum(sitk.GetArrayFromImage(lung_image) == 1)
        all_count = left_lung_count + right_lung_count
        print("left_lung_count/right_lung_count:",left_lung_count/right_lung_count)
        print("right_lung_count/left_lung_count:",right_lung_count/left_lung_count)
        if (left_lung_count/right_lung_count >= 1  and left_lung_count/right_lung_count <= 1.5) or (right_lung_count/left_lung_count >= 1  and right_lung_count/left_lung_count <= 1.5):
            return "Double_lung"
        else:
            return "Single_lung"

    def seg_double_lung(self,img_out_itk, tumor_path, input_image, mask_save, spacing, file_name, output_path):
        tumor_mask = sitk.ReadImage(tumor_path)
        # 计算肺部mask与肿瘤mask的交集
        lung_array = sitk.GetArrayFromImage(img_out_itk).astype('u1')
        tumor_array = sitk.GetArrayFromImage(tumor_mask).astype('u1')
        # lung_mask = np.zeros_like(lung_array)
        lung_mask = lung_array + tumor_array
        lung_mask[lung_mask > 0] = 1
        lung_image = sitk.GetImageFromArray(lung_mask)
        lung_image = self.fill_hole(lung_image)
        lung_image = self.remove_mixedTag(lung_image)
        lung_image.SetSpacing(spacing)
        self.multiply(lung_image, input_image, mask_save)
        sitk.WriteImage(lung_image, os.path.join(output_path, file_name + '_seg.nii.gz'))

    def read_xml_value(self,xml_file,value):
        """"
        读取xml节点内容
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # 获取<lung_Splite>节点的值
        lung_splite = root.find(value).text
        return lung_splite

    def seg_single_lung(self,lung_mask_sitk,tumor_path,input_image,mask_save,spacing,file_name,output_path):
        tumor_mask = sitk.ReadImage(tumor_path)
        # 计算肺部mask与肿瘤mask的交集
        # intersection = sitk.And(lung_mask_sitk, tumor_mask_sitk)
        lung_array = sitk.GetArrayFromImage(lung_mask_sitk).astype('u1')
        tumor_array = sitk.GetArrayFromImage(tumor_mask).astype('u1')
        intersection = lung_array * tumor_array
        # 统计左肺和右肺交集的像素数量
        left_lung_intersection_count = np.sum(
            intersection[sitk.GetArrayFromImage(lung_mask_sitk) == 2])
        right_lung_intersection_count = np.sum(
            intersection[sitk.GetArrayFromImage(lung_mask_sitk) == 1])
        # 判断交集像素数量，保存相应的肺部mask
        # 创建右肺mask和左肺mask
        right_lung_mask = np.zeros_like(lung_array)
        left_lung_mask = np.zeros_like(lung_array)
        if left_lung_intersection_count > right_lung_intersection_count:
            left_lung_mask[lung_array == 2] = 1
            left_lung_mask= left_lung_mask + tumor_array
            left_lung_mask[left_lung_mask > 0] = 1
            left_lung = sitk.GetImageFromArray(left_lung_mask)
            left_lung = self.fill_hole(left_lung)
            left_lung = self.remove_mixedTag(left_lung)
            left_lung.SetSpacing(spacing)
            self.multiply(left_lung, input_image, mask_save)
            sitk.WriteImage(left_lung, os.path.join(output_path,file_name + '_seg.nii.gz'))
            print("交集区域是左肺，已保存左肺mask")
        else:
            right_lung_mask[lung_array == 1] = 1
            right_lung_mask = right_lung_mask + tumor_array
            right_lung_mask[right_lung_mask > 0] = 1
            right_lung = sitk.GetImageFromArray(right_lung_mask)
            right_lung = self.fill_hole(right_lung)
            right_lung = self.remove_mixedTag(right_lung)
            right_lung.SetSpacing(spacing)
            self.multiply(right_lung, input_image, mask_save)
            sitk.WriteImage(right_lung, os.path.join(output_path,file_name + '_seg.nii.gz'))
            print("交集区域是右肺，已保存右肺mask")




















if __name__ == '__main__':


    # path = os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))
    # xmlfilepath = os.path.join(path, "test.txt")

    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument('method',
                        choices=['lung_segmentation', 'tumor_segmentation', "needle_segmentation", "rigid_registration",
                                 "remove_tumor", "norigid_registration", "difference_image", "difference_add",
                                 "caculate_tumor_center", "caculate_distance", "tumor_save","distance_hd",
                                 "abnormal_erase","statistical_value","cov_ratio","set_origin_to_zero"],
                        help='The method to call')
    parser.add_argument('-I', '--input_path', type=str,
                        default="None", help='the path to the root of dataset folders')
    parser.add_argument('-O', '--output_path', type=str,
                        default="None", help='the path to the root of dataset folders')
    parser.add_argument('-D', '--dataset_name', type=str,
                        default='None', help='the name of the dataset')
    parser.add_argument('-m', '--modality', type=str,
                        default='CT', help='the model of the dataset')
    parser.add_argument('-after_tumor_mask_file', '--after_tumor_mask_file', type=str,
                        default='None', help='the path of after_tumor_mask file')
    parser.add_argument('-middle_tumor_mask_file', '--middle_tumor_mask_file', type=str,
                        default='None', help='the path of middle_tumor_mask_file')
    parser.add_argument('-before_tumor_mask_file', '--before_tumor_mask_file', type=str,
                        default='None', help='the path of before_tumor_mask_file')
    parser.add_argument('-surgeryStatus', '--surgeryStatus', type=str,
                        default='None', help='the name of the surgeryStatus')
    parser.add_argument('-fixed_image', '--fixed_image', type=str,
                        default='None', help='the path of of the fixed image')
    parser.add_argument('-moving_image', '--moving_image', type=str,
                        default='None', help='the path of of the moving_image')
    parser.add_argument('-moving_image_mask', '--moving_image_mask', type=str,
                        default='None', help='the path of of the moving image mask')
    parser.add_argument('-fixed_mask ', '--fixed_mask', type=str,
                        default='None', help='the path of of the fixed mask')
    parser.add_argument('-moving_mask', '--moving_mask', type=str,
                        default='None', help='the path of of the moving mask')
    parser.add_argument('-moved_mask', '--moved_mask', type=str,
                        default='None', help='the path of of the moved mask')
    parser.add_argument('-after_tumor_mask ', '--after_tumor_mask', type=str,
                        default='None', help='the path of of the after tumor mask')
    parser.add_argument('-middle_tumor_mask ', '--middle_tumor_mask', type=str,
                        default='None', help='the path of of the middle tumor mask')
    parser.add_argument('-regid_image', '--regid_image', type=str,
                        default='None', help='regid lung image')
    parser.add_argument('-regid_tumor', '--regid_tumor', type=str,
                        default='None', help='regid tumor mask')
    parser.add_argument('-moved_image', '--moved_image', type=str,
                        default='None', help='regid lung image')
    parser.add_argument('-norigidtumor_mask', '--norigidtumor_mask', type=str,
                        default='None', help='noregid tumor mask')
    parser.add_argument('-norigidmoved_image', '--norigidmoved_image', type=str,
                        default='None', help='noregid image of lung')
    parser.add_argument('-difference_result', '--difference_result', type=str,
                        default='None', help='difference_result')
    parser.add_argument('-maskFile_before', '--maskFile_before', type=str,
                        default='None', help='maskFile_before')
    parser.add_argument('-difference_addmask', '--difference_addmask', type=str,
                        default='None', help='maskFile_before')
    parser.add_argument('-point1', '--point1', type=list,
                        help='point1')
    parser.add_argument('-point2', '--point2', type=list,
                        help='point2')
    parser.add_argument('-raw_mask', '--raw_mask', type=str,
                        default='None', help='raw_mask')
    parser.add_argument('-raw_image', '--raw_image', type=str,
                        default='None', help='raw_image')
    parser.add_argument('-save_path', '--save_path', type=str,
                        default='None', help='save_path')
    parser.add_argument('-lung_segPath', '--lung_segPath', type=str,
                        default='None', help='raw lung mask')




    args = parser.parse_args()
    # data_type = args.data_type
    task_root = os.path.join(os.path.abspath(args.output_path), args.dataset_name)
    output_path = args.output_path
    input_path = args.input_path
    dataset_name = args.dataset_name

    before_tumor_mask_file = args.before_tumor_mask_file
    after_tumor_mask_file = args.after_tumor_mask_file
    middle_tumor_mask_file = args.middle_tumor_mask_file

    surgeryStatus = args.surgeryStatus
    regidImage = args.regid_image
    regidTumorMask = args.regid_tumor
    fixed_image = args.fixed_image
    moving_image = args.moving_image
    moving_image_mask = args.moving_image_mask
    fixedAfterTumor_mask = args.fixed_mask
    movingBeforeTumor_mask = args.moving_mask
    moved_mask = args.moved_mask
    moved_image = args.moved_image
    norigidTumor_mask = args.norigidtumor_mask
    norigidMoved_image = args.norigidmoved_image
    difference_result = args.difference_result
    maskFile_before = args.maskFile_before
    difference_addmask = args.difference_addmask
    p1 = args.point1
    p2 = args.point2
    raw_mask = args.raw_mask
    raw_image = args.raw_image
    save_path = args.save_path
    modality = args.modality
    lung_segPath = args.lung_segPath





    breath = Breath()
    if args.method == 'lung_segmentation':
        if before_tumor_mask_file != "None":
            tumor_mask_file = before_tumor_mask_file
        elif middle_tumor_mask_file !="None":
            tumor_mask_file = middle_tumor_mask_file
        else:
            tumor_mask_file = after_tumor_mask_file
        breath.lung_segmentation(input_path, output_path,modality,tumor_mask_file,lung_segPath)
    elif args.method == "tumor_segmentation":
        breath.tumor_segmentation(input_path,output_path,dataset_name)
    elif args.method == "needle_segmentation":
        breath.needle_segmentation(input_path,dataset_name)
    elif args.method == "rigid_registration":
        if before_tumor_mask_file != "None":
            tumor_mask_file = before_tumor_mask_file
        elif middle_tumor_mask_file !="None":
            tumor_mask_file = middle_tumor_mask_file
        else:
            tumor_mask_file = after_tumor_mask_file
        breath.rigid_registration(args.fixed_image, args.moving_image, moving_image_mask, tumor_mask_file,dataset_name,surgeryStatus)
    elif args.method == "remove_tumor":
        try:
            breath.remove_tumor(regidImage,regidTumorMask,surgeryStatus,dataset_name)
        except RuntimeError as err:
            print("The tumor and needle does not exist:",err)
    elif args.method == "norigid_registration":
        try:
            breath.norigid_registration(fixed_image,moving_image,fixedAfterTumor_mask,movingBeforeTumor_mask,moved_mask,dataset_name)
        except RuntimeError as err:
            print(err,"The tumor too smoll ,please change image!")
    elif args.method == "difference_image":
        difference_result = breath.difference_image(moved_image,norigidTumor_mask,norigidMoved_image,dataset_name)
    elif args.method == "difference_add":
        breath.difference_add(difference_result,maskFile_before,dataset_name)
    elif args.method == "caculate_tumor_center":
        breath.caculate_tumor_center(input_path,dataset_name)
    elif args.method == "caculate_distance":
        breath.caculate_distance(p1,p2)
    elif args.method == "tumor_save":
        breath.tumor_save(input_path,maskFile_before,output_path)
    elif args.method =="distance_hd":
        breath.dis_bor(norigidTumor_mask,after_tumor_mask_file,dataset_name)
    elif args.method == "abnormal_erase":
        breath.abnormal_erase(raw_mask,raw_image,save_path,dataset_name,surgeryStatus)
    elif args.method == "cov_ratio":
        if middle_tumor_mask_file != "None":
            tumor_mask_file = middle_tumor_mask_file
        elif after_tumor_mask_file !="None":
            tumor_mask_file = after_tumor_mask_file
        else:
            tumor_mask_file = before_tumor_mask_file
        breath.cov_ratio(tumor_mask_file,norigidTumor_mask,dataset_name)
    elif args.method == "statistical_value":
        breath.statistical_value(after_tumor_mask_file)
    elif args.method == "set_origin_to_zero":
        breath.set_origin(input_path,output_path)


