# BoFiCap
NLPCC2023 Best Paper: Bounding and Filling: A Fast and Flexible Framework for Image Captioning

### Environment

1. We use conda to build virtual environment, and we export the env configs to `env.yaml`, you can reproduce the project env by this yaml.
2. Besides virtual env, evaluation relies on some metric project, so we build our project based on [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch), you can refer to this project to install metric project, e.g., cider.



### Data

1. We use MSCOCO Dataset as our dataset, and follow the standard Karpathy Splits.
2. To reproduce our training, you need to preprocess sentence in data into phrase level, and the preprocessed data had been uploaded in `data` folder.
3. We also provide preprocess scripts to build phrase datasets, the detailed usage will be upload soon.
4. AT LAST, your data folder should contain three files: 
   1. cocotalk_stanza_kd100_syn_dep0.json 
   2. cocotalk_stanza_kd100_syn_dep0_label.h5 
   3. [cocobu_att.lmdb](https://drive.google.com/file/d/1hun0tsel34aXO4CYyTRIvHJkcbZHwjrD/view?usp=sharing)  (detailed info refer to [here](https://github.com/ruotianluo/self-critical.pytorch/blob/master/data/README.md#image-features-option-2-bottom-up-features-current-standard).)



### Training

Our training process including two stages: Cross Entropy Training and Self Critical Training

1. Cross Entroy Training

   ```shell
   python tools/train.py --cfg configs/uic_sd.yaml --id any_thing_you_like
   ```

2. Self Critical Training

   ```shell
   python tools/train.py --cfg configs/uic_sd_kd100_sd_nscl.yaml --id any_thing_you_like
   ```

​	remember to edit the checkpoint path while you using self-critical training

### Test

```shell
python tools/eval.py --input_json data/cocotalk_stanza_kd100_syn_dep0.json --input_att_dir data/cocobu_att.lmdb --input_label_h5 data/cocotalk_stanza_kd100_syn_dep0_label.h5 --num_images -1 --model model.pth --infos_path infos.pkl --language_eval 0
```



