from __future__ import absolute_import, division, print_function
import argparse
import sys

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
import unet
from utils import normalize_image
from torch.utils.tensorboard.writer import SummaryWriter

writers = {}
cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def normalize(opt, depth_map):
    ma = opt.max_depth
    mi = opt.min_depth
    d = ma - mi if ma != mi else 1e5
    return (depth_map - mi) / d
    
def denormalize(opt, depth_map):
    ma = opt.max_depth
    mi = opt.min_depth
    d = ma - mi if ma != mi else 1e5
    return  mi + (depth_map * d) 


def evaluate(opt):
    print("####################################")
    print()
    print(opt.load_weights_folder)
    print()
    print("####################################")
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        high_encoder_path = os.path.join(opt.load_weights_folder, "encoder_high.pth")
        low_encoder_path = os.path.join(opt.load_weights_folder, "encoder_low.pth")
        high_decoder_path = os.path.join(opt.load_weights_folder, "depth_high.pth")
        
        img_ext = '.png' if opt.png else '.jpg'
        
        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           opt.high_height, opt.high_width, opt.low_height,opt.low_width,
                                           [0], 1, is_train=False, img_ext=img_ext)
        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        if opt.backbone in ["resnet", "resnet_lite"]:
            encoder_high = networks.ResnetEncoderDecoder(num_layers=opt.num_layers, num_features=opt.num_features, model_dim=opt.model_dim)
            encoder_low = networks.ResnetEncoderDecoder(num_layers=opt.num_layers, num_features=opt.num_features, model_dim=opt.model_dim)

        if opt.backbone.endswith("_lite"):
            depth_decoder_high = networks.Lite_Depth_Decoder_QueryTr(in_channels=opt.model_dim, patch_size=opt.patch_size_high, dim_out=opt.dim_out_high, embedding_dim=opt.model_dim, 
                                                                    query_nums=opt.query_nums_high, num_heads=4, min_val=opt.min_depth, max_val=opt.max_depth)
        #high
        high_encoder_dict = torch.load(high_encoder_path)
        model_dict_high = encoder_high.state_dict()
        encoder_high.load_state_dict({k: v for k, v in high_encoder_dict.items() if k in model_dict_high})        
        depth_decoder_high.load_state_dict(torch.load(high_decoder_path))

        encoder_high.cuda()
        encoder_high = torch.nn.DataParallel(encoder_high)
        encoder_high.eval()
        depth_decoder_high.cuda()
        depth_decoder_high = torch.nn.DataParallel(depth_decoder_high)
        depth_decoder_high.eval()
        
        #low
        low_encoder_dict = torch.load(low_encoder_path)
        model_dict_low = encoder_low.state_dict()
        encoder_low.load_state_dict({k: v for k, v in low_encoder_dict.items() if k in model_dict_low})        

        encoder_low.cuda()
        encoder_low = torch.nn.DataParallel(encoder_low)
        encoder_low.eval()
        
        
        #merge
        merge_net_path = os.path.join(opt.load_weights_folder, "merge.pth")
        merge_net=unet.UNet(64,32)
        merge_net.load_state_dict(torch.load(merge_net_path))
        merge_net.cuda()
        merge_net = torch.nn.DataParallel(merge_net)
        merge_net.eval()
        
        
        pred_disps = []
        src_imgs = []
        error_maps = []

        print("-> Computing predictions with size {}x{} & {}x{}".format(
             opt.high_height, opt.high_width, opt.low_height,opt.low_width))

        step = 0
        with torch.no_grad():
            for data in dataloader:
                
                step = step + 1
                input_color_high_out = data[("color", 0, 0)].cuda()
                input_color_low = data[("color_low", 0, 0)].cuda()
                
                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color_high = torch.cat((input_color_high_out, torch.flip(input_color_high_out, [3])), 0)
                    input_color_low = torch.cat((input_color_low, torch.flip(input_color_low, [3])), 0)
                    
                features_high = encoder_high(input_color_high)
                features_low = encoder_low(input_color_low)
                
                _, _, features_h, features_w = features_high.size()
                features_low = torch.nn.functional.interpolate(features_low,(features_h,features_w),mode='bilinear',align_corners=False) 
                outputs_merged = torch.cat((features_high,features_low), 1)
                features_merged = merge_net(outputs_merged)  
                
                output_high_out = depth_decoder_high(features_merged)["disp", 0]
                
                outputs = {}
                outputs["disp", 0] = output_high_out
                
               
                                
                
                

                pred_disp = outputs[("disp", 0)]
                # pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
                # src_imgs.append(data[("color", 0, 0)])
                
                # viz
        #         import PIL.Image as pil
        #         from PIL import ImageFile
        #         ImageFile.LOAD_TRUNCATED_IMAGES = True
        #         import matplotlib as mpl
        #         from matplotlib import pyplot as plt
        #         import matplotlib.cm as cm
                
        #         final = torch.nn.functional.interpolate(
        #         torch.unsqueeze(torch.from_numpy(pred_disp),dim=0), (384, 1280), mode="bilinear", align_corners=False)
        #         final = final.squeeze().cpu().numpy()
                
        #         high = torch.nn.functional.interpolate(
        #         output_high_out, (384, 1280), mode="bilinear", align_corners=False)
        #         high = high.cpu()[:, 0].numpy()
        #         N = high.shape[0] // 2
        #         high = batch_post_process_disparity(high[:N], high[N:, :, ::-1]).squeeze()
                
        #         low = torch.nn.functional.interpolate(
        #         output_low_out, (384, 1280), mode="bilinear", align_corners=False)
        #         low = low.cpu()[:, 0].numpy()
        #         N = low.shape[0] // 2
        #         low = batch_post_process_disparity(low[:N], low[N:, :, ::-1]).squeeze()
                
        #         output_directory = "/mnt/RG/SfMNeXt-Impl/viz"
        #         output_dir=data["path"][0][0]
        #         output_name = data["path"][1][0]
        #         to_save_dir = os.path.join(output_directory, output_dir,output_name)
                
                
        #         if not os.path.exists(to_save_dir):
        #             os.makedirs(to_save_dir)
                
        #         vmax = np.percentile(final, 95)
        #         normalizer = mpl.colors.Normalize(vmin=final.min(), vmax=vmax)
        #         mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma_r')
        #         # mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')
        #         colormapped_im = (mapper.to_rgba(final)[:, :, :3] * 255).astype(np.uint8)
        #         im = pil.fromarray(colormapped_im)
        #         name_dest_im = os.path.join(to_save_dir, "merged.jpeg")
        #         im.save(name_dest_im)
                
                
        #         colormapped_im = (mapper.to_rgba(high)[:, :, :3] * 255).astype(np.uint8)
        #         im = pil.fromarray(colormapped_im)
        #         name_dest_im = os.path.join(to_save_dir, "high.jpeg")
        #         im.save(name_dest_im)
                
        #         colormapped_im = (mapper.to_rgba(low)[:, :, :3] * 255).astype(np.uint8)
        #         im = pil.fromarray(colormapped_im)
        #         name_dest_im = os.path.join(to_save_dir, "low.jpeg")
        #         im.save(name_dest_im)
                
        #         input_color_high_out = input_color_high_out.squeeze().permute(1, 2, 0).cpu().numpy() * 255
        #         input_color_high_out = input_color_high_out.astype(np.uint8)
                
        #         im = pil.fromarray(input_color_high_out)
        #         name_dest_im = os.path.join(to_save_dir, "RGB.jpeg")
                
        #         im.save(name_dest_im)
                

        pred_disps = np.concatenate(pred_disps)
        # src_imgs = np.concatenate(src_imgs)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)
        # src_imgs_path = os.path.join(
        #     opt.load_weights_folder, "src_{}_split.npy".format(opt.eval_split))
        # print("-> Saving src imgs to ", src_imgs_path)
        # np.save(src_imgs_path, src_imgs)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        error_map = np.abs(gt_depth - pred_depth)
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        error_map = np.multiply(error_map, mask)
        error_maps.append(error_map)

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    if opt.save_pred_disps:
        error_map_path = os.path.join(
            opt.load_weights_folder, "error_{}_split.npy".format(opt.eval_split))
        print("-> Saving error maps to ", error_map_path)
        np.savez_compressed(error_map_path, data=np.array(error_maps, dtype="object"))
    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)

if __name__ == "__main__":
    options = MonodepthOptions()
    options.parser.convert_arg_line_to_args = convert_arg_line_to_args
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        opt = options.parser.parse_args([arg_filename_with_prefix])
    else:
        opt = options.parser.parse_args()
    writers["vis"] = SummaryWriter(os.path.join(opt.log_dir, "vis"))
    
    weight_list = [i for i in range(25)]
    for i in weight_list:
        opt.load_weights_folder = f"/mnt/RG/SfMNeXt-Impl/boost/resnet50_boost_RE_early/models/weights_{i}"
        evaluate(opt)
    # evaluate(options.parse())

