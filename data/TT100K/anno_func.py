import json
import pylab as pl
import random
import numpy as np
import cv2
import copy
import os
import math

type45="i2,i4,i5,il100,il60,il80,io,ip,p10,p11,p12,p19,p23,p26,p27,p3,p5,p6,pg,ph4,ph4.5,ph5,pl100,pl120,pl20,pl30,pl40,pl5,pl50,pl60,pl70,pl80,pm20,pm30,pm55,pn,pne,po,pr40,w13,w32,w55,w57,w59,wo"
type45 = type45.split(',')

def load_img(annos, datadir, imgid):
    img = annos["imgs"][imgid]
    imgpath = datadir+'/'+img['path']
    imgdata = pl.imread(imgpath)
    #imgdata = (imgdata.astype(np.float32)-imgdata.min()) / (imgdata.max() - imgdata.min())
    if imgdata.max() > 2:
        imgdata = imgdata/255.
    return imgdata

def load_mask(annos, datadir, imgid, imgdata):
    img = annos["imgs"][imgid]
    mask = np.zeros(imgdata.shape[:-1])
    mask_poly = np.zeros(imgdata.shape[:-1])
    mask_ellipse = np.zeros(imgdata.shape[:-1])
    for obj in img['objects']:
        box = obj['bbox']
        cv2.rectangle(mask, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])), 1, -1)
        if obj.has_key('polygon') and len(obj['polygon'])>0:
            pts = np.array(obj['polygon'])
            cv2.fillPoly(mask_poly, [pts.astype(np.int32)], 1)
            # print pts
        else:
            cv2.rectangle(mask_poly, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])), 1, -1)
        if obj.has_key('ellipse'):
            rbox = obj['ellipse']
            rbox = ((rbox[0][0], rbox[0][1]), (rbox[1][0], rbox[1][1]), rbox[2])
            print rbox
            cv2.ellipse(mask_ellipse, rbox, 1, -1)
        else:
            cv2.rectangle(mask_ellipse, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])), 1, -1)
    mask = np.multiply(np.multiply(mask,mask_poly),mask_ellipse)
    return mask

def get_instance_segs(boxes, annos, imgid, line_num=14):
    eps = 1e-3
    mask = load_mask(annos, "", imgid, img)

    all_seg_instance = []
    for box in boxes:
        IoU = []
        objs = annos["imgs"][imgid]["objects"]
        for obj in objs:
            bbox = [obj["bbox"]["xmin"], obj["bbox"]["ymin"], obj["bbox"]["xmax"], obj["bbox"]["ymax"]]
            IoU.append( calc_iou(box, bbox) )

        # find ground truth
        max_IoU_idx = np.argmax(IoU, axis=0)
        max_IoU = np.max(IoU, axis=0)
        if max_IoU<=0.1:
            continue

        obj = objs[max_IoU_idx]
        box_y1 = box[1]
        box_y2 = box[3]
        box_height = box_y2 - box_y1
        assert box_height>0

        bar_intervel = box_height / float(line_num)
        y_pos = []
        for k in xrange(line_num):
            y_pos.append( (k+0.5)*bar_intervel + box_y1 )

        # get segments
        bbox_x_seg = []
        polygon_x_seg = []
        ellipse_x_seg = []

        x1 = obj["bbox"]['xmin']
        x2 = obj["bbox"]['xmax']
        y1 = obj["bbox"]['ymin']
        y2 = obj["bbox"]['ymax']

        for y in y_pos:
            if y>=y1 and y<=y2:
                bbox_x_seg.append([x1,x2])
            else:
                bbox_x_seg.append([None, None])

            # polygon
            if obj.has_key('polygon') and len(obj['polygon'])>0:
                pts = obj['polygon']
                pts_num = len(pts)
                x_list = []

                for k in xrange(pts_num):
                    pt1 = pts[k]
                    pt2 = pts[(k+1)%pts_num]
                    y1 = pt1[1]
                    y2 = pt2[1]

                    if (y1-y)*(y2-y)<=0:
                        if pt1[0]<pt2[0]:
                            pt_left  = pt1
                            pt_right = pt2
                        else:
                            pt_left  = pt2
                            pt_right = pt1

                        if np.abs(y1-y2)<eps:
                            x_list.append(pt1[0])
                            x_list.append(pt2[0])
                        else:
                            x = pt_left[0] + (y-pt_left[1])*(pt_right[0]-pt_left[0])/(pt_right[1]-pt_left[1])
                            x_list.append(x)
                if len(x_list)==0:
                    polygon_x_seg.append([None, None])
                elif len(x_list)==1:
                    polygon_x_seg.append([x_list[0], x_list[0]])
                else:
                    x_left  = np.min(x_list)
                    x_right = np.max(x_list)
                    polygon_x_seg.append([x_left, x_right])
            else:
                polygon_x_seg.append([None, None])

            if obj.has_key('ellipse'):
                rbox = obj['ellipse']
                theta = rbox[2]*math.pi/180.0
                a = rbox[1][0]/2.0
                b = rbox[1][1]/2.0
                assert a>0 and b>0

                x_ctr = rbox[0][0]
                y_ctr = rbox[0][1]
                y_ = y - y_ctr
                A = np.cos(theta)*np.cos(theta)/(a*a) + np.sin(theta)*np.sin(theta)/(b*b)
                B = (1.0/(b*b) - 1.0/(a*a)) * y_ * np.sin(2*theta)
                C = (np.sin(theta)*np.sin(theta)/(a*a) + np.cos(theta)*np.cos(theta)/(b*b)) * y_*y_ - 1.0
                DELTA = B*B - 4*A*C

                assert A!=0
                if DELTA < 0:
                    ellipse_x_seg.append([None, None])
                else:
                    x1 = x_ctr + (-1*B - np.sqrt(DELTA))/(2.0*A)
                    x2 = x_ctr + (-1*B + np.sqrt(DELTA))/(2.0*A)
                    ellipse_x_seg.append([x1,x2])
            else:
                ellipse_x_seg.append([None, None])

        seg_ins = []
        for idx in xrange(len(y_pos)):
            y = y_pos[idx]

            xx1 = []
            xx2 = []
            if bbox_x_seg[idx][0]!=None:
                xx1.append(bbox_x_seg[idx][0])
                xx2.append(bbox_x_seg[idx][1])
            if polygon_x_seg[idx][0]!=None:
                xx1.append(polygon_x_seg[idx][0])
                xx2.append(polygon_x_seg[idx][1])
            if ellipse_x_seg[idx][0]!=None:
                xx1.append(ellipse_x_seg[idx][0])
                xx2.append(ellipse_x_seg[idx][1])
            if len(xx1)<1 or len(xx2)<1:
                continue

            x1 = np.max(xx1)
            x2 = np.min(xx2)

            mask_left  = None
            mask_right = None
            for _x in xrange(int(x1+0.5), int(x2+0.5)):
                if mask[int(y+0.5), _x]>0:
                    mask_left = _x
                    break
            for _x in xrange(int(x2+0.5), int(x1+0.5), -1):
                if mask[int(y+0.5), _x]>0:
                    mask_right = _x
                    break
            if mask_left==None and mask_right==None:
                x1 = 0
                x2 = 0
            elif mask_left!=None and mask_right==None:
                x1 = mask_left
                x2 = mask_left
            elif mask_right!=None and mask_left==None :
                x1 = mask_right
                x2 = mask_right
            else:
                x1 = np.max([x1, mask_left])
                x2 = np.min([x2, mask_right])
            seg_ins.append( [y,x1,x2])
        all_seg_instance.append(seg_ins)
    return all_seg_instance

def random_show_seg(annos, imgid, imgdata):
    objs = annos["imgs"][imgid]["objects"]

    rnd_boxes = []
    for obj in objs:
        box = [obj["bbox"]["xmin"], obj["bbox"]["ymin"], obj["bbox"]["xmax"], obj["bbox"]["ymax"]]
        rnd_num = np.random.uniform(-0.3, 0.3, [1,4])[0]
        box = box +  (box[3]-box[1])* rnd_num
        if box[2]<=box[0] or box[3]<=box[1]:
            continue
        rnd_boxes.append(box)

    # show
    segs = get_instance_segs(rnd_boxes, annos, imgid, line_num=14)
    for box in rnd_boxes:
        pt1 = (int(box[0]+0.5), int(box[1]+0.5))
        pt2 = (int(box[2]+0.5), int(box[3]+0.5))
        cv2.rectangle(imgdata, pt1, pt2, color=(0,0,255), thickness=3)

    for seg in segs:
        for y,x1,x2 in seg:
            pt1 = (int(x1+0.5), int(y+0.5))
            pt2 = (int(x2+0.5), int(y+0.5))
            cv2.line(imgdata, pt1, pt2, color=(0,255,0), thickness=1)
    return imgdata

def get_mask_x_pts(annos, imgid, img):
    img_ins = annos["imgs"][imgid]
    all_x_segs = []
    line_num = 14
    eps = 1e-3

    mask = load_mask(annos, "", imgid, img)
    for obj in img_ins['objects']:
        box = obj['bbox']
        x1 = box['xmin']
        x2 = box['xmax']
        y1 = box['ymin']
        y2 = box['ymax']

        box_height = y2-y1
        assert box_height>0

        # get y_pos
        bar_intervel = box_height / float(line_num)
        y_pos = []
        for k in xrange(line_num):
            y_pos.append( (k+0.5)*bar_intervel + y1 )

        # get segments
        bbox_x_seg = []
        polygon_x_seg = []
        ellipse_x_seg = []

        for y in y_pos:
            if y>=y1 and y<=y2:
                bbox_x_seg.append([x1,x2])
            else:
                bbox_x_seg.append([None, None])

            # polygon
            if obj.has_key('polygon') and len(obj['polygon'])>0:
                pts = obj['polygon']
                pts_num = len(pts)
                x_list = []

                for k in xrange(pts_num):
                    pt1 = pts[k]
                    pt2 = pts[(k+1)%pts_num]
                    y1 = pt1[1]
                    y2 = pt2[1]

                    if (y1-y)*(y2-y)<=0:
                        if pt1[0]<pt2[0]:
                            pt_left  = pt1
                            pt_right = pt2
                        else:
                            pt_left  = pt2
                            pt_right = pt1

                        if np.abs(y1-y2)<eps:
                            x_list.append(pt1[0])
                            x_list.append(pt2[0])
                        else:
                            x = pt_left[0] + (y-pt_left[1])*(pt_right[0]-pt_left[0])/(pt_right[1]-pt_left[1])
                            x_list.append(x)
                if len(x_list)==0:
                    polygon_x_seg.append([None, None])
                elif len(x_list)==1:
                    polygon_x_seg.append([x_list[0], x_list[0]])
                else:
                    x_left  = np.min(x_list)
                    x_right = np.max(x_list)
                    polygon_x_seg.append([x_left, x_right])
            else:
                polygon_x_seg.append([None, None])
	            
            if obj.has_key('ellipse'):
                rbox = obj['ellipse']
                theta = rbox[2]*math.pi/180.0
                a = rbox[1][0]/2.0
                b = rbox[1][1]/2.0
                assert a>0 and b>0

                x_ctr = rbox[0][0]
                y_ctr = rbox[0][1]
                y_ = y - y_ctr
                A = np.cos(theta)*np.cos(theta)/(a*a) + np.sin(theta)*np.sin(theta)/(b*b)
                B = (1.0/(b*b) - 1.0/(a*a)) * y_ * np.sin(2*theta)
                C = (np.sin(theta)*np.sin(theta)/(a*a) + np.cos(theta)*np.cos(theta)/(b*b)) * y_*y_ - 1.0
                DELTA = B*B - 4*A*C

                assert A!=0
                if DELTA < 0:
                    ellipse_x_seg.append([None, None])
                else:
                    x1 = x_ctr + (-1*B - np.sqrt(DELTA))/(2.0*A)
                    x2 = x_ctr + (-1*B + np.sqrt(DELTA))/(2.0*A)
                    ellipse_x_seg.append([x1,x2])
            else:
                ellipse_x_seg.append([None, None])
        # plot
        for idx in xrange(len(y_pos)):
            y = y_pos[idx]

            xx1 = []
            xx2 = []
            if bbox_x_seg[idx][0]!=None:
                xx1.append(bbox_x_seg[idx][0])
                xx2.append(bbox_x_seg[idx][1])
            if polygon_x_seg[idx][0]!=None:
                xx1.append(polygon_x_seg[idx][0])
                xx2.append(polygon_x_seg[idx][1])
            if ellipse_x_seg[idx][0]!=None:
                xx1.append(ellipse_x_seg[idx][0])
                xx2.append(ellipse_x_seg[idx][1])
            if len(xx1)<1 or len(xx2)<1:
                continue

            x1 = np.max(xx1)
            x2 = np.min(xx2)

            mask_left  = None
            mask_right = None
            for _x in xrange(int(x1+0.5), int(x2+0.5)):
                if mask[int(y+0.5), _x]>0:
                    mask_left = _x
                    break
            for _x in xrange(int(x2+0.5), int(x1+0.5), -1):
                if mask[int(y+0.5), _x]>0:
                    mask_right = _x
                    break
            if mask_left==None and mask_right==None:
                x1 = 0
                x2 = 0
            elif mask_left!=None and mask_right==None:
                x1 = mask_left
                x2 = mask_left
            elif mask_right!=None and mask_left==None :
                x1 = mask_right
                x2 = mask_right
            else:
                x1 = np.max([x1, mask_left])
                x2 = np.min([x2, mask_right])

            x1 = int(x1+0.5)
            x2 = int(x2+0.5)
            yy = int(y+0.5)

            cv2.line(img, (x1,yy), (x2,yy), (255,0,255), 1)
            cv2.circle(img, (x1,yy), 1, (0,255,0), 3)
            cv2.circle(img, (x2,yy), 1, (0,0,255), 3)
    return img
    

def draw_all(annos, datadir, imgid, imgdata, color=(0,1,0), have_mask=True, have_label=True):
    img = annos["imgs"][imgid]
    if have_mask:
        mask = load_mask(annos, datadir, imgid, imgdata)
        imgdata = imgdata.copy()
        imgdata[:,:,0] = np.clip(imgdata[:,:,0] + mask*0.7, 0, 1)
    for obj in img['objects']:
        box = obj['bbox']
        cv2.rectangle(imgdata, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])), color, 3)
        ss = obj['category']
        if obj.has_key('correct_catelog'):
            ss = ss+'->'+obj['correct_catelog']
        if have_label:
            cv2.putText(imgdata, ss, (int(box['xmin']),int(box['ymin']-10)), 0, 1, color, 2)
    return imgdata

def rect_cross(rect1, rect2):
    rect = [max(rect1[0], rect2[0]),
            max(rect1[1], rect2[1]),
            min(rect1[2], rect2[2]),
            min(rect1[3], rect2[3])]
    rect[2] = max(rect[2], rect[0])
    rect[3] = max(rect[3], rect[1])
    return rect

def rect_area(rect):
    return float(max(0.0, (rect[2]-rect[0])*(rect[3]-rect[1])))

def calc_cover(rect1, rect2):
    crect = rect_cross(rect1, rect2)
    return rect_area(crect) / rect_area(rect2)

def calc_iou(rect1, rect2):
    crect = rect_cross(rect1, rect2)
    ac = rect_area(crect)
    a1 = rect_area(rect1)
    a2 = rect_area(rect2)
    return ac / (a1+a2-ac)

def get_refine_rects(annos, raw_rects, minscore=20):
    cover_th = 0.5
    refine_rects = {}

    for imgid in raw_rects.keys():
        v = raw_rects[imgid]
        tv = copy.deepcopy(sorted(v, key=lambda x:-x[2]))
        nv = []
        for obj in tv:
            rect = obj[1]
            rect[2]+=rect[0]
            rect[3]+=rect[1]
            if rect_area(rect) == 0: continue
            if obj[2] < minscore: continue
            cover_area = 0
            for obj2 in nv:
                cover_area += calc_cover(obj2[1], rect)
            if cover_area < cover_th:
                nv.append(obj)
        refine_rects[imgid] = nv
    results = {}
    for imgid, v in refine_rects.items():
        objs = []
        for obj in v:
            mobj = {"bbox":dict(zip(["xmin","ymin","xmax","ymax"], obj[1])), 
                    "category":annos['types'][int(obj[0]-1)], "score":obj[2]}
            objs.append(mobj)
        results[imgid] = {"objects":objs}
    results_annos = {"imgs":results}
    return results_annos

def box_long_size(box):
    return max(box['xmax']-box['xmin'], box['ymax']-box['ymin'])

def eval_annos(annos_gd, annos_rt, iou=0.75, imgids=None, check_type=True, types=None, minscore=40, minboxsize=0, maxboxsize=400, match_same=True):
    ac_n, ac_c = 0,0
    rc_n, rc_c = 0,0
    if imgids==None:
        imgids = annos_rt['imgs'].keys()
    if types!=None:
        types = { t:0 for t in types }
    miss = {"imgs":{}}
    wrong = {"imgs":{}}
    right = {"imgs":{}}
    
    for imgid in imgids:
        v = annos_rt['imgs'][imgid]
        vg = annos_gd['imgs'][imgid]
        convert = lambda objs: [ [ obj['bbox'][key] for key in ['xmin','ymin','xmax','ymax']] for obj in objs]
        objs_g = vg["objects"]
        objs_r = v["objects"]
        bg = convert(objs_g)
        br = convert(objs_r)
        
        match_g = [-1]*len(bg)
        match_r = [-1]*len(br)
        if types!=None:
            for i in range(len(match_g)):
                if not types.has_key(objs_g[i]['category']):
                    match_g[i] = -2
            for i in range(len(match_r)):
                if not types.has_key(objs_r[i]['category']):
                    match_r[i] = -2
        for i in range(len(match_r)):
            if objs_r[i].has_key('score') and objs_r[i]['score']<minscore:
                match_r[i] = -2
        matches = []
        for i,boxg in enumerate(bg):
            for j,boxr in enumerate(br):
                if match_g[i] == -2 or match_r[j] == -2:
                    continue
                if match_same and objs_g[i]['category'] != objs_r[j]['category']: continue
                tiou = calc_iou(boxg, boxr)
                if tiou>iou:
                    matches.append((tiou, i, j))
        matches = sorted(matches, key=lambda x:-x[0])
        for tiou, i, j in matches:
            if match_g[i] == -1 and match_r[j] == -1:
                match_g[i] = j
                match_r[j] = i
                
        for i in range(len(match_g)):
            boxsize = box_long_size(objs_g[i]['bbox'])
            erase = False
            if not (boxsize>=minboxsize and boxsize<maxboxsize):
                erase = True
            #if types!=None and not types.has_key(objs_g[i]['category']):
            #    erase = True
            if erase:
                if match_g[i] >= 0:
                    match_r[match_g[i]] = -2
                match_g[i] = -2
        
        for i in range(len(match_r)):
            boxsize = box_long_size(objs_r[i]['bbox'])
            if match_r[i] != -1: continue
            if not (boxsize>=minboxsize and boxsize<maxboxsize):
                match_r[i] = -2
                    
        miss["imgs"][imgid] = {"objects":[]}
        wrong["imgs"][imgid] = {"objects":[]}
        right["imgs"][imgid] = {"objects":[]}
        miss_objs = miss["imgs"][imgid]["objects"]
        wrong_objs = wrong["imgs"][imgid]["objects"]
        right_objs = right["imgs"][imgid]["objects"]
        
        tt = 0
        for i in range(len(match_g)):
            if match_g[i] == -1:
                miss_objs.append(objs_g[i])
        for i in range(len(match_r)):
            if match_r[i] == -1:
                obj = copy.deepcopy(objs_r[i])
                obj['correct_catelog'] = 'none'
                wrong_objs.append(obj)
            elif match_r[i] != -2:
                j = match_r[i]
                obj = copy.deepcopy(objs_r[i])
                if not check_type or objs_g[j]['category'] == objs_r[i]['category']:
                    right_objs.append(objs_r[i])
                    tt+=1
                else:
                    obj['correct_catelog'] = objs_g[j]['category']
                    wrong_objs.append(obj)
                    
        
        rc_n += len(objs_g) - match_g.count(-2)
        ac_n += len(objs_r) - match_r.count(-2)
        
        ac_c += tt
        rc_c += tt
    if types==None:
        styps = "all"
    elif len(types)==1:
        styps = types.keys()[0]
    elif not check_type or len(types)==0:
        styps = "none"
    else:
        styps = "[%s, ...total %s...]"%(types.keys()[0], len(types))
    report = "iou:%s, size:[%s,%s), types:%s, accuracy:%s, recall:%s"% (
        iou, minboxsize, maxboxsize, styps, 1 if ac_n==0 else ac_c*1.0/ac_n, 1 if rc_n==0 else rc_c*1.0/rc_n)
    summury = {
        "iou":iou,
        "accuracy":1 if ac_n==0 else ac_c*1.0/ac_n,
        "recall":1 if rc_n==0 else rc_c*1.0/rc_n,
        "miss":miss,
        "wrong":wrong,
        "right":right,
        "report":report
    }
    return summury

if __name__=="__main__":
    fd = open("annotations.json", "r")
    anno_dict = json.load(fd)
    fd.close()

    img_root = "/data2/HongliangHe/work2017/TrafficSign/py-R-FCN/data/TT100K/train"
    for name in anno_dict["imgs"].keys():
        img_path = os.path.join(img_root, name.encode("utf-8")+".jpg")
        if not os.path.exists(img_path):
            print img_path,"not exists!"
            continue

        img = cv2.imread(img_path)

        #img = get_mask_x_pts(anno_dict, name, img)
        img = random_show_seg(anno_dict, name, img)
        img = cv2.resize(img, (1440,1440))
        cv2.imshow("img", img)
        cv2.waitKey(500)
