# encoding: utf-8
"""
@author: Yuke Lin
@contact: linyuke0609@gmail.com
"""
import os
import cv2
import torch
import numpy as np
from multiprocessing import Pool
from copy import deepcopy
from torchvision import transforms
from arcface import l2_norm, IResNet
class FaceRecognition():
    
    
    def __init__(self, path, device='cpu', mirror=False, mode ='ir'):
        
        self.device = device
        self.mirror = mirror
        self.mode = mode
        assert os.path.exists(path)
        self.path = path
        self.load_model(self.path)
       
       
    def load_model(self,path):
        if self.mode == 'resnet_v2':
            self.model = IResNet(model='res50')
            self.model.load_state_dict(torch.load(path))
            self.model = self.model.to(self.device)
        else:
            raise NotImplementedError
            
    def predict(self, img, meta=None):
        return self.predict_batch([img], [meta] if meta else None)[0]
    
    
    def predict_batch(self, imgs_list, meta_list=None):
        batch_data = self.prepare_batch_data(imgs_list, meta_list)
        batch_pred = self.compute_batch_data(batch_data)
        embd_list = []
        for embd in batch_pred:
            embd_list.append(embd)
        return embd_list

    
    def predict_video(self, video_path, dets_dict, batch_size):
        
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), 'Cannot open video file: {}'.format(video_path)
        
        buffer = []
        result = []
        for frame_idx in range(int(cap.get(7))):
            frame_idx = str(frame_idx)

            ret, img = cap.read()
            if not ret or frame_idx not in dets_dict:
                continue

            for meta in dets_dict[frame_idx]:
                buffer.append(dict(frame_idx=frame_idx, meta=deepcopy(meta), img=img.copy()))
                if len(buffer) >= batch_size:
                    result += self.compute_buffer(buffer)
                    buffer = []

        if len(buffer) > 0:
            result += self.compute_buffer(buffer)
            buffer = []

        cap.release()
        
        dets_dict = {}
        for data in result:
            frame_idx = data['frame_idx']
            if frame_idx not in dets_dict:
                dets_dict[frame_idx] = []
          
            dets_dict[frame_idx].append(data['meta']) 
            
        return dets_dict
    

    def prepare_single(self, image):
        face = cv2.resize(image, (112,112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = transforms.ToTensor()(face)
        face = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(face)
        return face
    

    def prepare_batch_data(self, imgs_list, meta_list,turbo=False):
        batch_data = []
        if meta_list == None and turbo:
            with Pool(processes=8) as pool:
                # MultiProcessing
                batch_data = pool.map(self.prepare_single, imgs_list)
        else:
            for i in range(len(imgs_list)):
                face = cv2.resize(imgs_list[i], (112,112))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = transforms.ToTensor()(face).to(self.device)
                face = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(face)
                batch_data.append(face)
        batch_data = torch.stack(batch_data, dim=0).float().to(self.device)
        return batch_data
        
        
    def compute_batch_data(self, batch_data, batch_size=1):
        total_batch = batch_data.size(0)

        self.model.eval()

        batch_pred_total = []

        with torch.no_grad():
            
            for i in range(0, total_batch, batch_size):
                sub_batch_data = batch_data[i:i+batch_size]

                sub_batch_pred = self.model(sub_batch_data)
                if self.mirror:
                    sub_batch_pred += self.model(sub_batch_data.flip(dims=[3]))
                sub_batch_pred = l2_norm(sub_batch_pred).detach().cpu().numpy()

                batch_pred_total.append(sub_batch_pred)

        batch_pred_total = np.concatenate(batch_pred_total, axis=0)

        return batch_pred_total
    
    
    def compute_buffer(self, buffer):
        
        imgs_list = [i['img'] for i in buffer]
        meta_list = [i['meta'] for i in buffer]
        embd_list = self.predict_batch(imgs_list=imgs_list, meta_list=meta_list)
        
        result = []
        for idx in range(len(buffer)):
            meta = buffer[idx]['meta']
            meta.update(dict(face_embd=embd_list[idx].tolist()))
            result.append(
                dict(frame_idx=buffer[idx]['frame_idx'], meta=meta)
            )
            
        return result