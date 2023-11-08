import argparse
import os
import re
import cv2
import time
import csv
import numpy as np
import pylab as p
import torch
import torch.nn.functional as F
from Models.RatNetAttention_DOConv import Net_ResnetAttention_DOConv

import pandas as pd
from pathlib import Path

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# filepath = "E:/***/***/***"
#
# filepath_list = []
# filename_list = []
# i = 0
# for file in Path(filepath).rglob('*.mp4'):
#     i += 1
#     filepath_list.append(file)
# # print(filepath_list)
# a=np.array(filepath_list)
# print(a.shape)
#
# for File in os.listdir(filepath):
#     if os.path.splitext(File)[1] == '.mp4': 
#         filename = File.split('.')[0]  
#         i += 1
#         filename_list.append(filename)
# # print(filename_list)
# b=np.array(filename_list)
# print(b.shape)


def get_args():
    parser = argparse.ArgumentParser(description='Predict keypoints from input images',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='models.pth',
                    metavar='FILE', help="Specify the file in which the model is stored")
    parser.add_argument('-i', '--input',
                    default='E:/***/***/',          
                    metavar='INPUT', nargs='+', help='filenames of input images')
    parser.add_argument('-o', '--output', default= 'E:/***/***/',  
                    metavar='OUTPUT', nargs='+', help='Filenames of output images')
    return parser.parse_args()

def preprocess(resize_w, resize_h, pil_img):
    pil_img = cv2.resize(pil_img, (resize_w, resize_h))
    img_nd = np.array(pil_img)
    if len(img_nd.shape) == 2:
        img_nd0 = img_nd
        img_nd = np.expand_dims(img_nd0, axis=2)
        img_nd = np.concatenate([img_nd, img_nd, img_nd], axis = -1)
    img_nd = img_nd.transpose((2, 0, 1))
    if img_nd.max() > 1:
        img_nd = img_nd / 255
    return img_nd


def predict_img(net,
                full_img,
                device,
                resize_w,
                resize_h,
                ):
    net.eval()

    img = torch.from_numpy(preprocess(resize_w, resize_h, full_img))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        probs = F.softmax(output, dim=1)

        probs = probs.squeeze(0)
        output = probs.cpu()
    return output



def heatmap_to_points(i, Img, heatmap, numPoints, ori_W, ori_H, keyPoints, keyPoints_all):
    for j in range(numPoints):
        hm = cv2.resize(heatmap[j, :, :], (ori_W, ori_H))
        center = np.unravel_index(np.argmax(hm), hm.shape)
        cx = center[1]
        cy = center[0]
        if i > 1:
            dis = np.sqrt((cx-keyPoints_all[i-1, 0, j])**2 + (cy-keyPoints_all[i-1, 1, j])**2)
            if dis < 70:
                keyPoints[0, j] = cx
                keyPoints[1, j] = cy
            else:
                delt = keyPoints_all[i - 1, :, :] - keyPoints_all[i - 2, :, :]
                delt = np.mean(delt, axis=1)
                keyPoints[:, j] = keyPoints_all[i - 1, :, j] + delt
        else:
            keyPoints[0, j] = cx
            keyPoints[1, j] = cy
        cv2.circle(Img, (int(keyPoints[0, j]), int(keyPoints[1, j])), 2, (0, 0, 255), 4)
    return Img, keyPoints

def draw_relation(Img, allPoints, relations):
    for k in range(len(relations)):
        c_x1 = int(allPoints[0, relations[k][0]-1])
        c_y1 = int(allPoints[1, relations[k][0]-1])
        c_x2 = int(allPoints[0, relations[k][1]-1])
        c_y2 = int(allPoints[1, relations[k][1]-1])
        cv2.line(Img, (c_x1, c_y1), (c_x2, c_y2), (0, 255, 0), 2)
    return Img


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    # in_files = in_files[0]
    out_files = args.output
    # out_files = out_files[0]
    excel_path = out_files + 'result.csv'
    npy_path = out_files + 'result.npy'
    print(in_files, out_files)

    isExists = os.path.exists(out_files)
    if not isExists:
        os.makedirs(out_files)

    keypoints = ['rRP', 'lRP', 'rFP', 'lFP', 'tail_root', 'head', 'neck', 'spine']
    relation = [[1, 5], [1, 8], [1, 7], [2, 5], [2, 7], [2, 8], [3, 7], [3, 6],
                [4, 6], [4, 7], [5, 8], [7, 8]]
    num_points = 10
    resize_w = 320
    resize_h = 256
    extract_list = ["layer4"]

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')           
    device = torch.device('cpu')
    net = Net_ResnetAttention_DOConv('none', extract_list, device, train=False, n_channels=3,
                                     nof_joints=num_points)

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device), strict=False)

    for filename in os.listdir(in_files):                           
        if filename.endswith(".mp4"):  # .avi or mp4
          
            f_path = os.path.join(in_files, filename)

            print(f_path)
       
            searchContext1 = '.'
            numList = [m.start() for m in re.finditer(searchContext1, filename)]
            out_files_video = out_files + filename[0: len(filename) - 4] + '/'

            isExists = os.path.exists(out_files_video)
            if not isExists:
                os.makedirs(out_files_video)
            excel_path = out_files_video + str(filename[0: len(filename) - 4]) +'result.csv'         
            npy_path = out_files_video + str(filename[0: len(filename) - 4]) + 'result.npy'
            saved_path = out_files_video + str(filename[0: len(filename) - 4]) + 'result2.avi'


            cap = cv2.VideoCapture(f_path)
            ori_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ori_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print('frames:', fps, ' total:', num_frames, ' size:', ori_height, ori_width)

            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            # saved_path = out_files + 'result.avi'
            video_writer = cv2.VideoWriter(saved_path, fourcc, 30, (640, 480))

            with open(excel_path, "a", newline='') as datacsv:
                csvwriter = csv.writer(datacsv, dialect="excel")
                csvwriter.writerow(['frame_num',
                                    'rRP_x', 'rRP_y',
                                    'lRP_x', 'lRP_y',
                                    'rFP_x', 'rFP_y',
                                    'lFP_x', 'lFP_y',
                                    'tail_x', 'tail_y',
                                    'head_x', 'head_y',
                                    'neck_x', 'neck_y',
                                    'spine_x',  'spine_y',
                                    'mean_x', 'mean_y'])

            i = 0
            # keyPoints_all = []
            keyPoints_all = np.zeros([int(num_frames), 2, num_points-2])
            keyPoints = np.zeros([2, num_points-2])
            while i < num_frames:
                ret, frame0 = cap.read()
                img = cv2.resize(frame0, (640, 480))
                Frame = img

                time_start = time.time()
                heatmap = predict_img(net=net,
                                      full_img=img,
                                      device=device,
                                      resize_w=resize_w,
                                      resize_h=resize_h)

                heatmap = heatmap.numpy().reshape((num_points, resize_h//4, resize_w//4))
                Frame, keyPoints = heatmap_to_points(i, Frame, heatmap, num_points-2, 640, 480, keyPoints, keyPoints_all)
                # cv2.imshow('img', frame0)
                # cv2.waitKey(0)
                Frame = draw_relation(Frame, keyPoints, relation)
                mean_x = np.mean(keyPoints[0, :])
                mean_y = np.mean(keyPoints[1, :])
                cv2.circle(Frame, (int(mean_x), int(mean_y)), 2, (255, 0, 0), 4)
                time_end = time.time()
                if i % 100 == 0:
                    print('time:', time_end-time_start, i)
                img_name = out_files + str(i) + '.png'



                cv2.putText(Frame, str(i), (620, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)        
                video_writer.write(Frame)
                # keyPoints_all.append(keyPoints)
                keyPoints_all[i, :, :] = keyPoints

                with open(excel_path, "a", newline='') as datacsv:
                    csvwriter = csv.writer(datacsv, dialect="excel")
                    csvwriter.writerow([i, keyPoints[0][0], keyPoints[1][0],
                                        keyPoints[0][1], keyPoints[1][1],
                                        keyPoints[0][2], keyPoints[1][2],
                                        keyPoints[0][3], keyPoints[1][3],
                                        keyPoints[0][4], keyPoints[1][4],
                                        keyPoints[0][5], keyPoints[1][5],
                                        keyPoints[0][6], keyPoints[1][6],
                                        keyPoints[0][7], keyPoints[1][7],
                                        mean_x, mean_y])
                i = i + 1

            # keyPoints_all = np.array(keyPoints_all)
            np.save(npy_path, keyPoints_all)
            print(keyPoints_all[199,:,:])