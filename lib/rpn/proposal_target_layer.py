# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
import sys
import os
import json
sys.path.append("../../data/TT100K")
from anno_func import get_instance_segs,random_show_seg
import pdb
import cv2

DEBUG = False

class ProposalTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self._num_classes = layer_params['num_classes']
        self._seg_num = layer_params['seg_num']
        self._anno_path = layer_params['anno_path']
        if not os.path.exists(self._anno_path):
            print self._anno_path,"not exits!"
            assert 0

        fd = open(self._anno_path, "r")
        self._anno_dict = json.load(fd)
        fd.close()

        # sampled rois (0, x1, y1, x2, y2)
        top[0].reshape(1, 5, 1, 1)
        # labels
        top[1].reshape(1, 1, 1, 1)
        # bbox_targets
        top[2].reshape(1, self._num_classes * 4, 1, 1)
        # bbox_inside_weights
        top[3].reshape(1, self._num_classes * 4, 1, 1)
        # bbox_outside_weights
        top[4].reshape(1, self._num_classes * 4, 1, 1)
        # segments targets
        top[5].reshape(1, self._seg_num*2+2, 1, 1)
        # segments weights
        top[6].reshape(1, self._seg_num*2+2, 1, 1)

    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0].data
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_boxes = bottom[1].data
        gt_boxes = gt_boxes.reshape(gt_boxes.shape[0], gt_boxes.shape[1])
        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported'

        rois_per_image = np.inf if cfg.TRAIN.BATCH_SIZE == -1 else cfg.TRAIN.BATCH_SIZE
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        # Sample rois with classification labels and bounding box regression
        # targets
        # print 'proposal_target_layer:', fg_rois_per_image
        labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes)

        '''if DEBUG:
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))'''

        # sampled rois
        # modified by ywxiong
        rois = rois.reshape((rois.shape[0], rois.shape[1], 1, 1))
        top[0].reshape(*rois.shape)
        top[0].data[...] = rois

        # classification labels
        # modified by ywxiong
        labels = labels.reshape((labels.shape[0], 1, 1, 1))
        top[1].reshape(*labels.shape)
        top[1].data[...] = labels

        # bbox_targets
        # modified by ywxiong
        bbox_targets = bbox_targets.reshape((bbox_targets.shape[0], bbox_targets.shape[1], 1, 1))
        top[2].reshape(*bbox_targets.shape)
        top[2].data[...] = bbox_targets

        # bbox_inside_weights
        # modified by ywxiong
        bbox_inside_weights = bbox_inside_weights.reshape((bbox_inside_weights.shape[0], bbox_inside_weights.shape[1], 1, 1))
        top[3].reshape(*bbox_inside_weights.shape)
        top[3].data[...] = bbox_inside_weights

        # bbox_outside_weights
        # modified by ywxiong
        bbox_inside_weights = bbox_inside_weights.reshape((bbox_inside_weights.shape[0], bbox_inside_weights.shape[1], 1, 1))
        top[4].reshape(*bbox_inside_weights.shape)
        top[4].data[...] = np.array(bbox_inside_weights > 0).astype(np.float32)

        # bottom[2] --> image_id, flip_flag
        #pdb.set_trace()
        assert bottom[2].data.size==2

        img_info = bottom[2].data.reshape([2])
        imgid = "%d"% int(img_info[0])
        flipped = bool(img_info[1])
        #print "proposal layer: image_id=",imgid,"flipped=",flipped

        # 1.Check the format of rois. 2.The format of segs
        ori_img_size = 2048
        now_img_size = cfg.TRAIN.SCALES[0]
        scale = np.float(now_img_size) / np.float(ori_img_size)
        roi_boxes = (rois[:,1:,0,0] * 1.0/scale).tolist()
        segs = get_instance_segs(roi_boxes, self._anno_dict, imgid, flipped, line_num=self._seg_num, img_size=2048)
        seg_targets = []
        seg_weights = []

        if DEBUG:
            img_root = "/data2/HongliangHe/work2017/TrafficSign/seg_mask/py-R-FCN/data/TT100K/images/"
            debug_save_path = "/data2/HongliangHe/work2017/TrafficSign/seg_mask/py-R-FCN/experiments/0509_seg_mask/debug_imgs"
            img_name = imgid+".jpg"
            img_path = os.path.join(img_root, img_name)
            img = cv2.imread(img_path)
            if flipped:
                #pdb.set_trace()
                img = cv2.flip(img, 1)

            for seg_ins in segs:
                if len(seg_ins)>0:
                    seg,start_pos,end_pos = seg_ins
                else:
                    continue
                for box in gt_boxes:
                    box = box[0:4] / scale
                    pt1 = (int(box[0]+0.5), int(box[1]+0.5))
                    pt2 = (int(box[2]+0.5), int(box[3]+0.5))
                    cv2.rectangle(img, pt1, pt2, color=(255,0,255), thickness=1)

                for y,x1,x2 in seg:
                    pt1 = (int(x1+0.5), int(y+0.5))
                    pt2 = (int(x2+0.5), int(y+0.5))
                    cv2.line(img, pt1, pt2, color=(0,255,0), thickness=1)
                    cv2.circle(img, pt1, 1, (255,0,0), 3)
                    cv2.circle(img, pt2, 1, (0,0,255), 3)

            save_path = os.path.join(debug_save_path, img_name)
            #img = cv2.resize(img, (1024,1024))
            cv2.imwrite(save_path, img)
            print "saved image:",save_path

        # if outside the image??
        for idx in xrange(len(segs)):
            seg_ins = segs[idx]
            if len(seg_ins) > 0:
                seg,start_pos,end_pos = seg_ins
                #print "start_pos:",start_pos,"end_pos:",end_pos
            else:
                seg = []
                start_pos = 0
                end_pos   = 0

            left_pts  = []
            right_pts = []
            left_weights  = []
            right_weights = []

            if len(seg)==0:
                for y_idx in xrange(self._seg_num):
                    left_pts.append(0)
                    left_weights.append(0)
                    right_pts.append(0)
                    right_weights.append(0)
            else:
                # ROIs (0, x1, y1, x2, y2)
                seg = np.array(seg) * scale
                roi_width = (rois[idx, 3] - rois[idx, 1])[0][0]
                roi_x_ctr = (rois[idx,3] + rois[idx,1])[0][0] /2.0

                for y_idx in xrange(self._seg_num):
                    if y_idx>=start_pos and y_idx<=end_pos:
                        x1 = seg[y_idx][1]
                        x2 = seg[y_idx][2]
                        dx1 = (x1 - roi_x_ctr) / roi_width
                        dx2 = (x2 - roi_x_ctr) / roi_width
                        left_pts.append(dx1)
                        left_weights.append(1.0)
                        right_pts.append(dx2)
                        right_weights.append(1.0)
                    else:
                        left_pts.append(0)
                        left_weights.append(0)
                        right_pts.append(0)
                        right_weights.append(0)

            dstart = np.float(start_pos) / np.float(self._seg_num)
            dend   = np.float(end_pos) / np.float(self._seg_num)
            #print "dstart:",dstart,"dend:",dend

            left_pts.extend(right_pts)
            left_pts.append( dstart )    # start_pos
            left_pts.append( dend )      # end_pos

            left_weights.extend(right_weights)
            left_weights.append( 2.0 )
            left_weights.append( 2.0 )

            seg_targets.append( left_pts )
            seg_weights.append( left_weights )

        # N * seg_num
        seg_targets = np.array(seg_targets).astype(np.float32)
        seg_targets = seg_targets.reshape((seg_targets.shape[0], seg_targets.shape[1], 1, 1))
        top[5].reshape(*seg_targets.shape)
        top[5].data[...] = np.array(seg_targets).astype(np.float32)

        seg_weights = np.array(seg_weights).astype(np.float32)
        seg_weights = seg_weights.reshape((seg_weights.shape[0], seg_weights.shape[1], 1, 1))
        top[6].reshape(*seg_weights.shape)
        top[6].data[...] = seg_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    # print 'proposal_target_layer:', bbox_targets.shape
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    if cfg.TRAIN.AGNOSTIC:
        for ind in inds:
            cls = clss[ind]
            start = 4 * (1 if cls > 0 else 0)
            end = start + 4
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    else:
        for ind in inds:
            cls = clss[ind]
            start = 4 * cls
            end = start + 4
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # print 'proposal_target_layer:', keep_inds
    
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]
    
    # print 'proposal_target_layer:', rois
    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    # print 'proposal_target_layer:', bbox_target_data
    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights
