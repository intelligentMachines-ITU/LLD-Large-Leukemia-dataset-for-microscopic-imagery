# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Loss functions."""

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel
from torch.nn.functional import cosine_similarity
import numpy as np 
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    xywh2xyxy,
    xyxy2xywh,
    non_max_suppression_ps,
    scale_boxes,
    get_fixed_xyxy,
    
)


def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441"""
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        """Initializes a modified BCEWithLogitsLoss with reduced missing label effects, taking optional alpha smoothing
        parameter.
        """
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """Computes modified BCE loss for YOLOv5 with reduced missing label effects, taking pred and true tensors,
        returns mean loss.
        """
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true,mask, object_loss,patch):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        
        loss = self.loss_fcn(pred, true)
        if object_loss:
             loss= loss*mask
           
                
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if object_loss:
            if patch: 
                if self.reduction == "mean":
                    # return loss.mean()
                    return loss.sum()/(mask > 0).sum().item()
                elif self.reduction == "sum":
                    return loss.sum()
            else: 
                if self.reduction == "mean":
                        return loss.mean()
                elif self.reduction == "sum":
                    return loss.sum()
        else:
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()

        return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes Quality Focal Loss with given loss function, gamma, alpha; modifies reduction to 'none'."""
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Computes the focal loss between `pred` and `true` using BCEWithLogitsLoss, adjusting for imbalance with
        `gamma` and `alpha`.
        """
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.stride = m.stride
        self.device = device
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)] 

    def __call__(self, p, targets, feat,epoch):  # predictions, targets
        """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""
        epsilon = 1e-12 
        p_lcls = torch.zeros(1, device=self.device)  # class loss
        p_lbox = torch.zeros(1, device=self.device)  # box loss
        p_lobj = torch.zeros(1, device=self.device)
        f_lcls = torch.zeros(1, device=self.device)  # class loss
        f_lbox = torch.zeros(1, device=self.device)  # box loss
        f_lobj = torch.zeros(1, device=self.device)
        
        # target= targets[:,:6]
        # masked_values_patch = torch.zeros(p, dtype=p.dtype, device=self.device)
        p_region= torch.cat((targets[:,0:1],targets[:,6:10]),dim=1)    
        p_n=[]
        complete_patch_mask= []
        for i in range(self.nl):
            selected_ratio=[8,16,32]
            # size_ratio=[(32,32),(16,16),(8,8)]
            
            
            unique_boxes, _ = torch.unique(p_region, dim=0, return_inverse=True)
            extracted_slices= []
            patch_mask=[]
            for l in range(p[i].shape[0]):
                _, x1, y1, x2, y2 = unique_boxes[l].int()
                x1, x2, y1, y2 = int(x1 / selected_ratio[i]), int(x2 / selected_ratio[i]), int(y1 / selected_ratio[i]), int(y2 / selected_ratio[i])

                # Create a mask of zeros
                mask = torch.zeros_like(p[i][l])
                
                # Set the specified region to 1
                mask[:, y1:y2, x1:x2, :] = 1

                # Apply the mask to p[i][l]
                masked_p = p[i][l] * mask
                
                
                
                
                # import cv2
        
                # import numpy as np
                # roi_aligned_features= masked_p[:,:,:,-1].detach().cpu().numpy()      
                # roi_aligned_features_numpy = roi_aligned_features
                # roi_aligned_features_numpy[roi_aligned_features_numpy < 0] *= -1     
                # roi_aligned_features_numpy = np.transpose(roi_aligned_features_numpy, (1, 2, 0))

                #                         # Write the NumPy array to an image file using OpenCV
                # processed_array = ((roi_aligned_features_numpy) * 255).astype(np.uint8)
                #         #processed_array = np.squeeze(processed_array)

                #         # Write the processed image using OpenCV
                # save_path = f"{l}_output.jpg"
                # cv2.imwrite(save_path, processed_array)
                
                
                
                
                
                
                
                
                
                

                # Extract the masked slice
                # extracted_slice = masked_p[:, y1:y2, x1:x2, :]
                
                extracted_slices.append(masked_p.squeeze(0))
                patch_mask.append(mask.squeeze(0))
        
            p_n.append(torch.stack(extracted_slices, dim=0))
            complete_patch_mask.append(torch.stack(patch_mask, dim=0))
        
        
        
       
        p_tcls, p_tbox, p_indices, p_anchors = self.build_targets(p_n, targets) 
        
        p_lbox , p_lobj ,p_lcls, p_bs, t_class_all,p_classes_path= self. loss_com_patch(p_tcls, p_tbox, p_indices, p_anchors,p_n,complete_patch_mask)
        pseudo_targets, patch_targets,orignal_targets,pseudo_targets_60_90 = self.pseudo_targets(p, targets)
        
        mean_sim=0
        # # similarity_loss_patch_2= self.similarity(feat,pseudo_targets_60_90, patch_targets,orignal_targets,0 )
        # # if similarity_loss_patch_2 !=  torch.zeros(1, device=self.device):
        # #    similarity_loss_patch_2=  similarity_loss_patch_2.unsqueeze(0)
        # #    mean_sim+=1
        if epoch < 20 :
            similarity_loss_patch_3= self.similarity(feat,pseudo_targets_60_90, patch_targets,orignal_targets,1 )
            if similarity_loss_patch_3 !=  torch.zeros(1, device=self.device):
                similarity_loss_patch_3=  similarity_loss_patch_3.unsqueeze(0)
                mean_sim+=1
            if mean_sim > 0:
                
                
                
                
                
                
                
                
                
                
                
                
                
                similarity_loss_patch= (similarity_loss_patch_3)#+(similarity_loss_patch_3)/mean_sim
            else: 
                similarity_loss_patch= (similarity_loss_patch_3)#+(similarity_loss_patch_3)
        if epoch >= 20:
            similarity_loss_patch = torch.zeros(1, device=self.device)  # class loss
        # # similarity_loss_patch_4= self.similarity(feat,pseudo_targets_60_90, patch_targets,orignal_targets,2 )
        # # if similarity_loss_patch_4 !=  torch.zeros(1, device=self.device):
        # #     similarity_loss_patch_4=  similarity_loss_patch_4.unsqueeze(0)
        # #     mean_sim+=1
        
        if len(pseudo_targets) > 0: 
            f_tcls, f_tbox, f_indices, f_anchors = self.build_targets(p, pseudo_targets) 
            f_lbox , f_lobj ,f_lcls, p_classes= self. loss_com_background(f_tcls, f_tbox, f_indices, f_anchors,p)
            
        # i_tcls, i_tbox, i_indices, i_anchors = self.build_targets(p, targets) 
        
        # # i_class_all= self.loss_com_image(i_tcls, i_tbox, i_indices, i_anchors,p)
        
        
        
        
        
        # # kl_loss= ((self.compute_kl_loss(t_class_all,p_classes_path))*0.2) + epsilon
        
        
        
        #sim_loss
        
        
        
        
        
        
        if len(pseudo_targets) > 0: 
            lbox= p_lbox + (f_lbox*0.1)
            lobj= p_lobj +(f_lobj*0.1)
            lcls= (p_lcls)+(f_lcls*0.1)+(similarity_loss_patch*0.1) #+ (kl_loss)
            # lcls= similarity_loss_patch
            
        else: 
            lbox= p_lbox
            lobj= p_lobj
            lcls= p_lcls+(similarity_loss_patch*0.1)#+kl_loss
        
        if epoch == 29:
            lbox= p_lbox
            lobj= p_lobj
            lcls= p_lcls
        # if similarity_loss_patch !=  torch.zeros(1, device=self.device):
        #    similarity_loss_patch=  similarity_loss_patch.unsqueeze(0)
        # return (kl_loss) * p_bs, torch.cat((kl_loss.unsqueeze(0),p_lobj, p_lcls)).detach()
        return (lbox + lobj + lcls) * p_bs, torch.cat((p_lbox, p_lobj, p_lcls)).detach()
        # return (lbox + lobj + lcls) * p_bs, torch.cat((p_lbox, p_lobj, p_lcls,f_lbox, f_lobj, f_lcls, similarity_loss_patch)).detach()
    def loss_com_patch (self,tcls, tbox, indices, anchors,p_n,complete_patch_masks):
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss84653940
    
         # targets
        t_classes_path=[]
        p_classes_path=[]

        # Losses
        for i, (pi, pq) in enumerate(zip(p_n, complete_patch_masks)):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t,pcls,object_loss=False, patch= False)  # BCE
                # t_classes_path.append(t)
                # p_classes_path.append(pcls)
                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj,pq[..., 4],object_loss=True,patch= True)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
            

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size
        # sim_loss= torch.tensor(0.00, dtype=torch.float16, device='cuda:0')
        # kl_loss= torch.tensor(0.00, dtype=torch.float16, device='cuda:0')
        # torch.tensor(0.001, dtype=torch.float16, device='cuda:0')
        # t_classes_path= torch.cat(t_classes_path)
        # p_classes_path=torch.cat(p_classes_path)
        return lbox , lobj ,lcls, bs ,t_classes_path,p_classes_path
        
    def loss_com_background (self,tcls, tbox, indices, anchors,p):
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        p_classes_all=[]
         # targets
        # gt_path= paths[0].split("/")[-1].split(".")[0]
        # gt_part=gt_path.split("_")
        # gt_part[3]="1000"
        # del gt_part[2]
        # gt_path= '_'.join(gt_part)
        # x_prediction=np.load(f"100x_GT/{gt_path}.npy", allow_pickle=True) 
        
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh,pobj, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions
                
                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t ,pcls,object_loss=False,patch= False)  # BCE
                p_classes_all.append(pcls)

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
            # tobj[b, a, gj, gi] = iou  # iou ratio
            mask = torch.zeros_like(pi[..., 4], dtype=torch.bool)
            mask[b, a, gj, gi] = True

            # Apply the mask to pi[..., 4]
            masked_values = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)

            masked_values[mask] = pi[..., 4][mask]
                
                # masked_values[mask] = pi[..., 4][mask]
            obji = self.BCEobj(masked_values, tobj,mask,object_loss=True,patch= False)
            # obji*=
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                    self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        # lbox *= self.hyp["box"]
        # lobj *= self.hyp["obj"]
        # lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size
        # sim_loss= torch.tensor(0.00, dtype=torch.float16, device='cuda:0')
        # kl_loss= torch.tensor(0.00, dtype=torch.float16, device='cuda:0')
        # torch.tensor(0.001, dtype=torch.float16, device='cuda:0')
        p_classes_all= torch.cat(p_classes_all, dim=0)

        return lbox , lobj ,lcls, p_classes_all
    
    def loss_com_image (self,tcls, tbox, indices, anchors,p):
        image_classes_all=[]
        
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            # tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh,pobj, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions
                image_classes_all.append(pcls)
        image_classes_all= torch.cat(image_classes_all)

        return image_classes_all



    def build_targets(self, p, target):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        
        """
        
        targets= target[:,:6]
        
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets* gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

    
    
    
    
    def compute_kl_loss(self, tcls, pred):
                    # pcls_sigmoid = torch.sigmoid(pred)
                    tcls_sigmoid = tcls
                    data2 = self.gumbel_softmax(pred, tau=1, hard=False)

                    # Get the indices of the maximum values along each row
                    # max_indices = torch.argmax(pcls_sigmoid, dim=1)

                    # Create a new tensor using these indices
                    # new_tensor2 = max_indices.device
                    # new_tensor2=  new_tensor2.to(dtype=torch.float)
                    # new_tensor2.requires_grad_(True)

                    # Combine tensor values into a single list for both sets
                    # data1 = []
                    data1 = [torch.tensor(lst) for lst in tcls]

                    # Stack tensors along a new dimension (dimension 0 by default)
                    stacked_data1= torch.stack(data1)

                    # Perform element-wise sum along dimension 0
                    data1_counts = torch.sum(stacked_data1, dim=0).detach()
                    
                    # data1= []
                    # data2 = pcls_softmax  #new_tensor2.cpu().tolist()
                    # for tensor in tcls_sigmoid:
                    #     data1.extend(tensor.cpu().tolist())
                    # # for tensord in pcls_sigmoid:
                    # #     data2.extend(tensord.cpu().tolist())
                    

                    # # Convert to probability distributions
                    # data1_counts = np.bincount(data1, minlength=14)
                    # data1_counts = torch.sum(data1, dim=0)
                    data2_counts = torch.sum(data2, dim=0)
                    # data2_counts = np.bincount(data2, minlength=14)
                    
                    epsilon = 1e-12 

                    # Normalize to get probabilities
                    data1_probs = (data1_counts+epsilon)  / sum(data1_counts)
                    data2_probs = (data2_counts+epsilon)  / sum(data2_counts)

                    # Convert to tensors
                    data1_probs = torch.tensor(data1_probs, device='cuda:0', dtype=torch.float, requires_grad=True)
                    # data2_probs = torch.tensor(data2_probs, device='cuda:0', dtype=torch.float, requires_grad=True)

                    # Compute KL divergence
                    kl_div = F.kl_div(data1_probs.log(), data2_probs, reduction='batchmean')
                    return kl_div
    def gumbel_softmax(self,logits, tau=1, hard=False, eps=1e-10):
        gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim=-1)

        if hard:
            # Straight through
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparameterization trick
            ret = y_soft
        return ret
    
    def make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') #if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid
    def calculate_overlap(self,box1, box2):
        """Calculate the overlap area between two bounding boxes."""
        # Extracting coordinates of the intersection rectangl+e
        box1=box1*640
        box2=box2*640
        x_left = torch.max(box1[0] - box1[2] / 2, box2[0] - box2[2] / 2)
        y_top = torch.max(box1[1] - box1[3] / 2, box2[1] - box2[3] / 2)
        x_right = torch.min(box1[0] + box1[2] / 2, box2[0] + box2[2] / 2)
        y_bottom = torch.min(box1[1] + box1[3] / 2, box2[1] + box2[3] / 2)

        # Calculate width and height of the intersection rectangle
        width = torch.clamp(x_right - x_left, min=0)
        height = torch.clamp(y_bottom - y_top, min=0)

        # If the intersection is valid (non-negative area), return the area
        intersection_area = width * height
        return intersection_area
    def merge_tensors(self,tensor1, tensor_9):
        """Merge two tensors while keeping all values from tensor1 and discarding overlapped values from tensor2."""
        without_patch_tensor = []
        patch_tensor= []
        tensor2= tensor_9[:,:6]
        

        # Add all bounding boxes from tensor1
        # merged_tensor.extend(tensor1)

        # Iterate through each bounding box in tensor2
        for box2 in tensor2:
            overlap = False

            # Check for overlap with each bounding box in tensor1
            for box1 in tensor1:
             if  box1[0] == box2[0]:
                if self.calculate_overlap(box1[2:], box2[2:]) > 100:   
                    overlap = True
                    break

                # If there's no overlap, add the bounding box from tensor2
            if overlap:
                patch_tensor.append(box2)
            if not overlap:
                without_patch_tensor.append(box2)
        if without_patch_tensor: 
            without_patch_tensor = torch.stack(without_patch_tensor)
            without_patch_tensor = without_patch_tensor[without_patch_tensor[:, 0].argsort()]
        # else:
        #     merged_tensor= torch.tensor(merged_tensor).device

        return without_patch_tensor,patch_tensor





    def pseudo_targets(self, p, target):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        
        
        
        
        
        
        
        
        """
        
        # targets= target[:,:6]
        
        # p_region= torch.cat((target[:,0:1],target[:,6:10]),dim=1)    
        # p_n=[]
        # for i in range(self.nl):
        #     selected_ratio=[8,16,32]
        #     size_ratio=[(32,32),(16,16),(8,8)]
            
            
        #     unique_boxes, _ = torch.unique(p_region, dim=0, return_inverse=True)
        #     extracted_slices= []
        #     for l in range(p[i].shape[0]):
        #         _, x1, y1, x2, y2 = unique_boxes[l].int()
        #         x1, x2, y1, y2 = int(x1 / selected_ratio[i]), int(x2 / selected_ratio[i]), int(y1 / selected_ratio[i]), int(y2 / selected_ratio[i])

        #         # Create a mask of zeros
        #         mask = torch.ones_like(p[i][l])
                
        #         # Set the specified region to 1
        #         mask[:, y1:y2, x1:x2, :] = 0

        #         # Apply the mask to p[i][l]
        #         masked_p = p[i][l] * mask

        #         # Extract the masked slice
        #         # extracted_slice = masked_p[:, y1:y2, x1:x2, :]
                
        #         extracted_slices.append(masked_p.squeeze(0))
        
        #     p_n.append(torch.stack(extracted_slices, dim=0))
        
        
        
        
        
        z=[]
        
        p_clone= p.copy()
        for i in range(self.nl):
            bs, self.na, ny, nx, self.no = p_clone[i].shape 
            
            xy, wh, conf = p_clone[i].sigmoid().split((2, 2, self.nc + 1), 4)
            self.grid[i], self.anchor_grid[i] = self.make_grid(nx, ny, i)
            xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
            wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
            y = torch.cat((xy, wh, conf), 4)
            z.append(y.view(bs, self.na * nx * ny, self.no))
        
        z_new= torch.cat(z, 1)

        lb = [targets_lb[targets_lb[:, 0] == i, 1:] for i in range(range(p_clone[0].shape[0]))] if False else []






        train_preds = non_max_suppression_ps(
                 z_new.detach().cpu(), 0.5, 0.2, labels=lb, multi_label=False, agnostic=True, max_det=300
             )
       
        # del sgrid
        # del anchor_grid
        
        train_pseudo_labels = [torch.tensor([]) for _ in range(len(train_preds))]
        train_pseudo_labels_60_90 = [torch.tensor([]) for _ in range(len(train_preds))]

        for num, preds in enumerate(train_preds):
            for bbox in preds:
                # print((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                if ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) > 10:
                    # print(bbox[4])
                    if bbox[4] > 0.95:

                        last_14_values = bbox[-14:].softmax(dim=0)

                        # Normalize the values to make them probabilities
                        probs = last_14_values / last_14_values.sum()

                        # Create a categorical distribution
                        dist = torch.distributions.Categorical(probs)

                        # Calculate entropy

                        entropy = dist.entropy()
                        # print(entropy)
                        if entropy.item() < 2.7: 
                            
                            bbox_tensor = torch.tensor(bbox).unsqueeze(0)  # Convert bbox to tensor and add a batch dimension
                            train_pseudo_labels[num] = torch.cat((train_pseudo_labels[num], bbox_tensor), dim=0)
                        # else:
                    elif 0.60 < bbox[4] <= 0.95:
                        bbox_tensor = torch.tensor(bbox).unsqueeze(0)

                        train_pseudo_labels_60_90[num] = torch.cat((train_pseudo_labels_60_90[num], bbox_tensor), dim=0)
                        
                        
        
                            # print(entropy)
        
        
        train_pseudo_box = [([])  for _ in range(len(train_pseudo_labels)) ]
        train_pseudo_box_60_90 = [[] for _ in range(len(train_pseudo_labels_60_90))]

        for num2, pred_box in enumerate(train_pseudo_labels):
            for bbox in pred_box:
                        # train_pseudo_box[num2].append(bbox[:4].detach().cpu().numpy())
                        train_pseudo_box[num2].append(bbox.detach().cpu().numpy())
        for num2, pred_box in enumerate(train_pseudo_labels_60_90):
            for bbox in pred_box:
                train_pseudo_box_60_90[num2].append(bbox.detach().cpu().numpy())
        
        
        pesudo_target_list = []
        pesudo_target_list_60_90 = []

        for i, tensor in enumerate(train_pseudo_labels):
            for t in tensor:
                tensor_values = [
                    i,
                    t[5].item(),  # [6]
                    torch.tensor(int(t[0].item())),  # [0]
                    torch.tensor(int(t[1].item())),  # [1]
                    torch.tensor(int(t[2].item())),  # [2]
                    torch.tensor(int(t[3].item()))   # [3]
                ]
                pesudo_target_list.append(tensor_values)
                
        for i, tensor in enumerate(train_pseudo_labels_60_90):
            for t in tensor:
                tensor_values = [
                    i,
                    t[5].item(),
                    torch.tensor(int(t[0].item())),
                    torch.tensor(int(t[1].item())),
                    torch.tensor(int(t[2].item())),
                    torch.tensor(int(t[3].item()))
                ]
                pesudo_target_list_60_90.append(tensor_values)

        # Convert the inner lists to tensors
        pesudo_target_list = [torch.tensor(tensor_values) for tensor_values in pesudo_target_list]
        pesudo_target_list_60_90 = [torch.tensor(tensor_values) for tensor_values in pesudo_target_list_60_90]
        targets_ps= target[:,:6]
        
                    
                    
                    # optimizer_cell_model.zero_grad(

        # Stack the tensors along a new dimension (dimension 0 in this example)
        wp_target=[]
        p_target=[]
        wp_target_60_90=[]
        
        if pesudo_target_list: 
            pesudo_target_list_concatenated = torch.stack(pesudo_target_list, dim=0)

            pesudo_target_list_concatenated= pesudo_target_list_concatenated/torch.tensor([1,1,640,640,640,640])
            
            x_min, y_min, x_max, y_max = pesudo_target_list_concatenated[:, 2], pesudo_target_list_concatenated[:, 3], pesudo_target_list_concatenated[:, 4], pesudo_target_list_concatenated[:, 5]

            # Calculating x_center, y_center, width, height
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            # Updating the tensor with new values
            pesudo_target_list_concatenated[:, 2] = x_center
            pesudo_target_list_concatenated[:, 3] = y_center
            pesudo_target_list_concatenated[:, 4] = width
            pesudo_target_list_concatenated[:, 5] = height
         
            
        
            
            pesudo_target_list_concatenated = pesudo_target_list_concatenated.to(target.device)
            wp_target,p_target  = self.merge_tensors(targets_ps,pesudo_target_list_concatenated)
        if pesudo_target_list_60_90:
            pesudo_target_list_concatenated_60_90 = torch.stack(pesudo_target_list_60_90, dim=0)
            pesudo_target_list_concatenated_60_90 = pesudo_target_list_concatenated_60_90 / torch.tensor([1, 1, 640, 640, 640, 640])
            
            x_min, y_min, x_max, y_max = pesudo_target_list_concatenated_60_90[:, 2], pesudo_target_list_concatenated_60_90[:, 3], pesudo_target_list_concatenated_60_90[:, 4], pesudo_target_list_concatenated_60_90[:, 5]
            
            # Calculating x_center, y_center, width, height
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            
            # Updating the tensor with new values
            pesudo_target_list_concatenated_60_90[:, 2] = x_center
            pesudo_target_list_concatenated_60_90[:, 3] = y_center
            pesudo_target_list_concatenated_60_90[:, 4] = width
            pesudo_target_list_concatenated_60_90[:, 5] = height
            
            pesudo_target_list_concatenated_60_90 = pesudo_target_list_concatenated_60_90.to(target.device)
            wp_target_60_90,p_target_60_90  = self.merge_tensors(targets_ps,pesudo_target_list_concatenated_60_90)
        # else:

        return wp_target, p_target, targets_ps,wp_target_60_90
    def calculate_iou(self,bbox1, bbox2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Arguments:
        bbox1 (tuple): Coordinates of the first bounding box in the format (x1, y1, x2, y2).
        bbox2 (tuple): Coordinates of the second bounding box in the format (x1, y1, x2, y2).

        Returns:
        float: Intersection over Union (IoU) between the two bounding boxes.
        """
        # Extract coordinates of the bounding boxes
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate the coordinates of the intersection rectangle
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        # If there's no intersection, return 0
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Calculate the area of intersection rectangle
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate the area of both bounding boxes
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate the area of union
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area
        
        return iou
    def xywh_to_xyxy(self,xywh):
                        x_center, y_center, width, height = xywh
                        x_min = x_center - width / 2
                        y_min = y_center - height / 2
                        x_max = x_center + width / 2
                        y_max = y_center + height / 2
                        return torch.tensor([x_min, y_min, x_max, y_max])
    def compute_similarity(self,feature_maps):
        num_samples = len(feature_maps)
        similarity_matrix = torch.zeros((num_samples, num_samples))
        
        # Apply average pooling to each feature map
        pooled_feature_maps = [F.avg_pool2d(fm, fm.shape[2]) for fm in feature_maps]
        
        for i in range(num_samples):
            for j in range(num_samples):
                # Calculate cosine similarity between pooled feature maps
                similarity_matrix[i, j] = F.cosine_similarity(pooled_feature_maps[i].view(-1), pooled_feature_maps[j].view(-1), dim=0)
        
        return similarity_matrix

                
    def feat_box(self,pseudo_targets,int_feat, Num_targets,p_layer ):
        pooled_feature_map_pseudo= []
        is_bbox=[]
        for i in range(Num_targets):
                img_num = int(pseudo_targets[i,0].item())

                p2_feature_map =int_feat[p_layer ][img_num]#pred[0][img_num][:,:,:,-1]# imgs[img_num] 
                            
                x_center = pseudo_targets[i, 2]
                y_center = pseudo_targets[i, 3]
                width = pseudo_targets[i, 4]
                height = pseudo_targets[i, 5]
                bb = [round(x_center.item(),4), round(y_center.item(),4), round(width.item(),4), round(height.item(),4)]
                p2_feature_shape_tensor = torch.tensor([p2_feature_map.shape[2], p2_feature_map.shape[1],p2_feature_map.shape[2],p2_feature_map.shape[1]])                        # reduce_channels_layer = torch.nn.Conv2d(1280, 250, kernel_size=1).to(device)
                            
                p2_normalized_xyxy = self.xywh_to_xyxy(bb)*p2_feature_shape_tensor #imgs.shape[2]
                            
                p2_x_min, p2_y_min, p2_x_max, p2_y_max = get_fixed_xyxy(p2_normalized_xyxy,p2_feature_map)
                            
                batch_index = torch.tensor([0], dtype=torch.float32).to(self.device)

                p2_roi = torch.tensor([p2_x_min, p2_y_min, p2_x_max, p2_y_max], device=self.device).float() 
                is_bbox.append(p2_roi)
                            # Concatenate the batch index to the bounding box coordinates
                p2_roi_with_batch_index = torch.cat([batch_index, p2_roi])
                            

                # relevant_feature_map = p3_feature_map.unsqueeze(0)[:, :, y_min:y_max, x_min:x_max]
                p2_resized_object = torchvision.ops.roi_align(p2_feature_map.unsqueeze(0), p2_roi_with_batch_index.unsqueeze(0).to(self.device), output_size=(4, 4))     
                pooled_feature_map_pseudo.append(p2_resized_object)
        return  pooled_feature_map_pseudo, is_bbox
    def similarity(self,int_feat,pseudo_targets, patch_targets, orignal_targets,p_layer):
        
        

                  
        losses = []
        Num_targets = len(pseudo_targets)
        Num_targets_label = len(orignal_targets)
        target_featuers, target_box= self.feat_box(orignal_targets,int_feat, Num_targets_label,p_layer)
        
        # for feat2, label2, is_bb in zip(target_featuers, orignal_targets[:,1],target_box):
                # similarity=0
        
            
                # for feat1, label1 , ps_bb in zip( target_featuers, orignal_targets[:,1],target_box ):
                #     iou= self.calculate_iou(ps_bb,is_bb)
                #     if iou < 0.3:
                #         feat2 = F.avg_pool2d(feat2, feat2.shape[2])
                #         if label1 == label2:
                            
                #             feat1 = F.avg_pool2d(feat1, feat1.shape[2])

                #             similarity =  cosine_similarity(feat1,feat2).mean()
                #             # similarity=(similarity+1)/2 
                #             losses.append(1-similarity)
                #         if label1 != label2:
                            
                #             feat1 = F.avg_pool2d(feat1, feat1.shape[2])
                            
                #             similarity =   cosine_similarity(feat1,feat2).mean()
                #             similarity=(similarity+1)/2 
                #             alpha= 0.2
                #             torch.max(torch.tensor(0.0), similarity )
                #             losses.append(similarity)
                            
                                
                                    
                            
                #                     # Calculate similarity between features (e.g., cosine similari
        
        
        
        
        
        if Num_targets :
            pseudo_featuers,pseudo_box= self.feat_box(pseudo_targets,int_feat, Num_targets,p_layer  )
            
    
            for feat2, label2, is_bb in zip(pseudo_featuers,  pseudo_targets[:,1],pseudo_box):
                #sim=-10
        
                min_same_class_sim=100
                max_diff_class_sim= 0
                for feat1, label1 , ps_bb in zip( target_featuers, orignal_targets[:,1],target_box ):
                    iou= self.calculate_iou(ps_bb,is_bb)
                    feat2 = F.avg_pool2d(feat2, feat2.shape[2])
                    feat1 = F.avg_pool2d(feat1, feat1.shape[2])
                                
                    if iou < 0.3:  
                                    
                            
                                    # Calculate similarity between features (e.g., cosine similarity)
                        similarity =   cosine_similarity(feat1,feat2).mean()
                        similarity=(similarity+1) # Normalize 
                        
                        if label1 == label2:
                            if min_same_class_sim > similarity:
                                min_same_class_sim =  similarity
                                
                        else:
                            if  max_diff_class_sim < similarity:
                                 max_diff_class_sim =  similarity
                        
                        
                       
                                        

                        # max_similarity=(max_similarity+1)/2                
                                    # Compare labels
                if min_same_class_sim < 100  and max_diff_class_sim > 0:
                    loss_value = torch.max(torch.tensor(0.0), min_same_class_sim - (max_diff_class_sim+0.05))
                    losses.append(loss_value)
                                                    # Compute loss (e.g., squared difference)
                                                # loss_sim = torch.mean(torch.abs(torch.tensor(feat1) - torch.tensor(feat2)))
                #                                 losses.append(max_similarity)
                # elif ((max_similarity < 0.30) and (max_label1 == max_label2.item())):
                #                                     # Compute loss (e.g., squared difference)
                #                                 # loss_sim = torch.mean(torch.abs(torch.tensor(feat1) - torch.tensor(feat2)))
                #                                 max_similarity=(max_similarity+1)/2
                #                                 losses.append(1-max_similarity)
                        
        
        if losses:
            total_loss = (torch.sum(torch.stack(losses))/len(losses) )
        else:
            total_loss= torch.zeros(1, device=self.device)
        return(total_loss)
                            
        #     for feat2, label2, is_bb in zip(pseudo_featuers,  pseudo_targets[:,1],pseudo_box):
        #         sim=-10
        
            
        #         for feat1_gr, label1 , ps_bb in zip( target_featuers, orignal_targets[:,1],target_box ):
        #             feat1 = feat1_gr.clone().detach()
        #             iou= self.calculate_iou(ps_bb,is_bb)
        #             feat2 = F.avg_pool2d(feat2, feat2.shape[2])
        #             feat1 = F.avg_pool2d(feat1, feat1.shape[2])
                                
        #             if iou < 0.3:  
                                    
                            
        #                             # Calculate similarity between features (e.g., cosine similarity)
        #                 similarity =   cosine_similarity(feat1,feat2).mean()
        #                 if similarity > sim:
        #                     sim=similarity
        #                     max_similarity = similarity 
        #                     max_label1 =label1
        #                     max_label2= label2
                                        

        #                 # max_similarity=(max_similarity+1)/2                
        #                             # Compare labels
        #                 if ((max_similarity  > 0.95) and (max_label1 != max_label2)):
        #                                             # Compute loss (e.g., squared difference)
        #                                         # loss_sim = torch.mean(torch.abs(torch.tensor(feat1) - torch.tensor(feat2)))
        #                                         losses.append(max_similarity)
        #                 elif ((max_similarity < 0.50) and (max_label1 == max_label2.item())):
        #                                             # Compute loss (e.g., squared difference)
        #                                         # loss_sim = torch.mean(torch.abs(torch.tensor(feat1) - torch.tensor(feat2)))
        #                                         max_similarity=(max_similarity+1)/2
        #                                         losses.append(1-max_similarity)
                        
        
        # if losses:
        #     total_loss = (torch.sum(torch.stack(losses))/len(losses) )
        # else:
        #     total_loss= torch.zeros(1, device=self.device)
        # return(total_loss)
                            
        
