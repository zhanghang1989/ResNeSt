## Train ResNeSt with MXNet Gluon

For training with PyTorch, please visit [PyTorch Encoding Toolkit](https://hangzhang.org/PyTorch-Encoding/model_zoo/imagenet.html)

### Install MXNet with Horovod

```bash
# assuming you have CUDA 10.0 on your machine
pip install mxnet-cu100
HOROVOD_GPU_ALLREDUCE=NCCL pip install -v --no-cache-dir horovod
pip install --no-cache mpi4py
```

### Prepare ImageNet recordio data format

- Unfortunately ,this is required for training using MXNet Gluon. Please follow the [GluonCV tutorial](https://gluon-cv.mxnet.io/build/examples_datasets/recordio.html) to prepare the data.
- Copy the data into ramdisk (optional):
	
	```
	cd ~/
	sudo mkdir -p /media/ramdisk
	sudo mount -t tmpfs -o size=200G tmpfs /media/ramdisk
	cp -r /home/ubuntu/data/ILSVRC2012/ /media/ramdisk
	```

### Training command

Using ResNeSt-50 as the target model:

```bash
horovodrun -np 64 --hostfile hosts python train.py \
--rec-train /media/ramdisk/ILSVRC2012/train.rec \
--rec-val /media/ramdisk/ILSVRC2012/val.rec \
--model resnest50 --lr 0.05 --num-epochs 270 --batch-size 128 \
--use-rec --dtype float32 --warmup-epochs 5 --last-gamma --no-wd \
--label-smoothing --mixup --save-dir params_ resnest50 \
--log-interval 50 --eval-frequency 5 --auto_aug --input-size 224
```

### Verify pretrained model

```bash
python verify.py --model resnest50 --crop-size 224 --resume params_ resnest50/imagenet-resnest50-269.params
```