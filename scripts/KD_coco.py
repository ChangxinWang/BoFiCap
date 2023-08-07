import json
import os

def main(ori_fp, kd_fp, tgt_fp):
    imgs = json.load(open(ori_fp, 'r'))
    imgs = imgs['images']
    new_imgs = imgs
    kd = json.load(open(kd_fp, 'r'))
    # print(len(kd.items()))
    # exit()
    bad_kd = 0
    cnt = 0
    for i, img in enumerate(imgs):
        if i % 5000 == 0:
            print(i)
        if img['split'] != 'train' and img['split'] != 'restval':
            continue
        cocoid = str(img['cocoid'])
        if cocoid not in kd:
            bad_kd += 1
            if bad_kd > 100:
                print("bad too much!")
                exit(0)
            continue
        ref = kd[cocoid]
        for j in range(1, 5):
            new_imgs[i]['sentences'][j]['raw'] = ref[j-1]
            new_imgs[i]['sentences'][j]['tokens'] = ref[j-1].split()
        cnt += 1
        if cnt % 2 == 1:
            new_imgs[i]['sentences'][0]['raw'] = ref[4]
            new_imgs[i]['sentences'][0]['tokens'] = ref[4].split()

    print("bad_kd : {}  train_cnt : {}".format(bad_kd, cnt))
    tgt = {'images':new_imgs}
    json.dump(tgt, open(tgt_fp, 'w'))

if __name__ == '__main__':
    ori_fp = 'data/dataset_coco.json'
    kd_fp = 'KD_dataset/0305.json'
    tgt_fp = 'data/dataset_coco_kd90.json'
    main(ori_fp, kd_fp, tgt_fp)
