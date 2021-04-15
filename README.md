# torch-demo
推荐使用单卡训练
推荐在单机多卡上使用ddp，dp的优化问题较大，在某些服务器上由于传输带宽低，将严重制约性能

# shell命令：
python -m torch.distributed.launch --nproc_per_node=8 step1_train_DDP.py
python -m torch.distributed.launch step1_train_DDP.py （在某些机器上可行）

# 参考：
https://fyubang.com/2019/07/23/distributed-training3/

https://blog.csdn.net/zwqjoy/article/details/89415933