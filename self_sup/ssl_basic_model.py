import numpy as np
import matplotlib.pyplot as plt
import os
from misc.imutils import save_image
from self_sup import *

import torch

import utils
from utils import de_norm
from misc.torchutils import norm_tensor

import cv2

# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class CDEvaluator():

    def __init__(self, args):

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args)

        self.device = torch.device("cuda:%s" % args.gpu_ids[0]
                                   if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")

        self.net_G.visualize = True  # 保存中间注意力图
        self.net_G.save_features = False  # 保留梯度图

        print(self.device)

        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = os.path.join(args.checkpoint_dir, 'vis')
        os.makedirs(self.vis_dir, exist_ok=True)
        self.pred_dir = os.path.join(args.checkpoint_dir, 'pred')
        os.makedirs(self.pred_dir, exist_ok=True)

    def load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name),
                                    map_location=self.device)

            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.net_G.to(self.device)

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)
        return self.net_G


    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.G_pred = self.net_G(img_in1, img_in2)
        return self._visualize_pred()

    def save_feat(self, feats, save_folder, names, suffix_name='x1_t1', feat_cat=False):
        from misc.torchutils import save_feature_maps
        from misc.imutils import save_image
        b, c, h, w = feats.shape
        for i in range(b):
            if feat_cat:  # 把各个维度的feat 拼成一图
                tensor_data = norm_tensor(feats[i])
                tensor_data = tensor_data.reshape([-1, 1, h, w])
                np_data = utils.make_numpy_grid(tensor_data, pad_value=0, padding=5)
                # save_feature_maps(save_folder, tensor_data, name_prefix='%s'%i)
                np_data = np.clip(np_data, a_min=0.0, a_max=1.0)
                att_map_color = cv2.applyColorMap((np_data * 255).astype(np.uint8), colormap=cv2.COLORMAP_JET)
                att_map_color = att_map_color[..., ::-1]
                save_image(att_map_color, save_folder+'/feat_%s_%s.jpg'
                           % (names[i], suffix_name))
            else:  # 挨个feat维度单独保存
                indexs = list(range(20))
                tensor_data = feats[i][indexs]
                tensor_data = norm_tensor(tensor_data)
                # out_folder = os.path.join(save_folder, 'feat_%s' % names[i])
                # os.makedirs(out_folder, exist_ok=True)
                save_feature_maps(save_folder, tensor_data, name_prefix='feat_%s'
                           % (suffix_name))

    # def save_imgs(self, ):

    def eval(self):
        self.net_G.eval()

    def _get_vis(self, batch_id=0):

        vis = self.net_G.vis
        self.net_G.vis = []

        features = torch.cat([norm_tensor(vis[0]), norm_tensor(vis[1])], dim=0)

        vis_feature = utils.make_numpy_grid_singledim(features, padding=5, pad_value=0)
        # vis_feature = np.sqrt(vis_feature)

        file_name = os.path.join(
            self.vis_dir, 'feat_' + self.batch['name'][0]+'.jpg')
        plt.imsave(file_name, vis_feature, cmap='jet')

        images = torch.cat([de_norm(self.batch['A']),
                             de_norm(self.batch['B'])], dim=0)

        vis = utils.make_numpy_grid(images, pad_value=255, padding=5)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(
            self.vis_dir, 'img_' + self.batch['name'][0]+'.jpg')
        plt.imsave(file_name, vis)

    def _get_cam(self):
        vis = self.net_G.vis
        self.net_G.vis = []

        c = vis[0].shape[1]
        features = torch.cat([norm_tensor(vis[0]), norm_tensor(vis[1])], dim=0)
        att_map = utils.make_numpy_grid_singledim(features, padding=5, pad_value=0)

        # att_map = np.sqrt(att_map)

        att_map_color = cv2.applyColorMap((att_map*255).astype(np.uint8), colormap=cv2.COLORMAP_JET)
        att_map_color = att_map_color[...,::-1]

        images = torch.cat([de_norm(self.batch['A']),]*c+[
                             de_norm(self.batch['B'])]*c, dim=0)

        vis = utils.make_numpy_grid(images, pad_value=255, padding=5)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        alpha = 0.7
        vis = vis * alpha + att_map_color/255*(1-alpha)
        file_name = os.path.join(
            self.vis_dir, '_cam_' + self.batch['name'][0]+'.jpg')
        plt.imsave(file_name, vis)

    def _save_predictions(self):
        preds = self._visualize_pred()
        name = self.batch['name']
        for i, pred in enumerate(preds):
            file_name = os.path.join(
                self.pred_dir, name[i].replace('.jpg', '.png'))
            pred = pred[0].cpu().numpy()
            save_image(pred, file_name)

    def _save_feats(self):
        f1 = self.net_G.f1
        f2 = self.net_G.f2
        target = torch.cat([1-self.batch['L'],self.batch['L']],dim=1).cuda()
        g = self.net_G.get_grad(target=target)

        index = torch.argmax(g, dim=1)
        index = range(32)
        f1_new = self.net_G.f1_new
        f2_new = self.net_G.f2_new

        f1_dif = norm_tensor(torch.abs(f1_new - f1))
        f2_dif = norm_tensor(torch.abs(f2_new - f2))

        vis_tensors1 = torch.cat([f1[:, index], f2[:, index]], dim=1)
        vis_tensors2 = torch.cat([f1_new[:, index], f2_new[:, index]], dim=1)
        vis_tensors3 = torch.cat([f1_dif[:, index], f2_dif[:, index]], dim=1)

        vis_feature = utils.make_numpy_grid_singledim(norm_tensor(vis_tensors1))
        name = self.batch['name']
        file_name = os.path.join(
            self.vis_dir, 'test_feat_old_' + name[0]+'.jpg')
        plt.imsave(file_name, vis_feature, cmap='jet')
        vis_feature = utils.make_numpy_grid_singledim(norm_tensor(vis_tensors2))
        file_name = os.path.join(
            self.vis_dir, 'test_feat_new_' + name[0]+'.jpg')
        plt.imsave(file_name, vis_feature, cmap='jet')
        vis_feature = utils.make_numpy_grid_singledim(norm_tensor(vis_tensors3))
        file_name = os.path.join(
            self.vis_dir, 'test_feat_dif_' + name[0]+'.jpg')
        plt.imsave(file_name, vis_feature, cmap='jet')

    def save_imgs(self, img, name):
        vis = utils.make_numpy_grid(img)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(
            self.vis_dir, "sample_"+name +'.jpg')
        plt.imsave(file_name, vis)

    def vis(self, batch_id):
        """save image/label/predictions """
        save_mode = 2
        if save_mode == 1:
            vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
            vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))
            vis_pred = utils.make_numpy_grid(self._visualize_pred())
            vis_gt = utils.make_numpy_grid(self.batch['L'])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
        else:
            tensors = torch.cat([de_norm(self.batch['A']),
                                 de_norm(self.batch['B']),
                                 torch.cat([self.batch['L'],]*3,dim=1),
                                 torch.cat([self._visualize_pred(), ] * 3, dim=1).cpu()],
                                dim=0)
            vis = utils.make_numpy_grid(tensors, pad_value=255)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(
            self.vis_dir, 'sample_' + str(batch_id)+'.jpg')
        plt.imsave(file_name, vis)

