#python3 -u train.py --momentum=0.9 --lr=1.0 --batch_size=80 --distill_mode=True|tee distill.log
python3 -u train.py --momentum=0.9 --lr=0.1 --batch_size=80 --distill_mode=True --resume=717906 --max_epoch=999999999|tee distill_continue.log
