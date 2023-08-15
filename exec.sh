## 需要对比单独训练以及整体训练的效果
## 三个头一起训练
python mtl/main.py /data2/rzhang/mtl_data \
    --epochs 200 \
    --train_head \
    --gpu 1
## 三个头单独训练
python mtl/main_separate.py /data2/rzhang/mtl_data \
    --epochs 200 \
    --head 1 \
    --gpu 1

python mtl/main_separate.py /data2/rzhang/mtl_data \
    --epochs 200 \
    --head 2 \
    --gpu 1 \
    --load_from work_dir/model_best_head1.pth

python mtl/main_separate.py /data2/rzhang/mtl_data \
    --epochs 200 \
    --head 3 \
    --gpu 1 \
    --load_from work_dir/model_best_head2.pth

############################################
###           特别注意顺序                 ###
###          涉黄 涉政 涉恐                ###
############################################

# python mtl/main.py /home/rzhang/data/MTL_data \
#     --epochs 200 \
#     --train_head \
#     --resume work_dir/model_best.pth \
#     --gpu 0


### export onnx
python mtl/test.py --task export \
    --ckpt work_dir_bak/model_best.pth

## predict images
python mtl/test.py --task predict \
    --ckpt work_dir/model_best.pth \
    --data test_data

### eval
python mtl/test.py --task eval \
    --ckpt work_dir_bak/model_best.pth \
    --gpu 1 \
    --data /data2/rzhang/mtl_data
### 三个头一起训练的精度
# 91.4944559733073
# 93.96001307169597
# 88.74327977498372