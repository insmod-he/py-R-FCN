import json
import os

if __name__=="__main__":
    root_path = "/data2/HongliangHe/work2017/TrafficSign/py-R-FCN/ImageSets/tt100k/"
    lab_path  = "annotations.json"
    lab_dict  = json.load(open( os.path.join(root_path, lab_path)))

    fds = []
    train_list = "train.txt"
    test_list  = "test.txt"
    other_list = "other.txt"
    fds.append(open(os.path.join(root_path, train_list), "w"))
    fds.append(open(os.path.join(root_path, test_list), "w"))
    #fds.append(open(os.path.join(root_path, other_list), "w"))

    all_lab_dict ={}
    img_dict = lab_dict["imgs"]
    for img_name in img_dict.keys():
        path = img_dict[img_name]["path"]
        data_set = path.split("/")[0]

        if data_set=="train":
            fd = fds[0]
        elif data_set=="test":
            fd = fds[1]
        #elif data_set=="other":
        #    fd = fds[2]
        print >>fd,img_name

        bbs = []
        cat = []
        for obj in img_dict[img_name]["objects"]:
            x1 = obj["bbox"]["xmin"]
            x2 = obj["bbox"]["xmax"]
            y1 = obj["bbox"]["ymin"]
            y2 = obj["bbox"]["ymax"]
            c  = obj["category"]
            bbs.append([x1,y1,x2,y2])
            cat.append(c)
        tmp_dict = {}
        tmp_dict["bbs"] = bbs
        tmp_dict["category"] = cat
        all_lab_dict[img_name] = tmp_dict

    # save
    fd = open( os.path.join(root_path, "all_lab_dict.json") ,"w")
    json.dump(all_lab_dict,fd)
    fd.close()
    print "done!"
