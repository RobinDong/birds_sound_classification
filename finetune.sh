#python3 train.py --momentum=0.9 --lr=100 --batch_size=128 --resume=3240000 --finetune=True --max_epoch=200|tee finetune.log
python3 train.py --momentum=0.9 --lr=0.2 --batch_size=400 --resume=877982 --finetune=True --max_epoch=9000000|tee finetune.log
