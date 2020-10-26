一、安装

conda create --name work1 python=3.7
conda activate work1
pip install tensorflow
pip install transformers --use-feature=2020-resolver
conda install tqdm pandas scikit-learn
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

二、TODO

数据清洗，比如在文本中，会有一些颜文字或者表情之类的，或许可以替换成正规的中文？

更换模型

改变训练方法，现在是k折训练，有没有别的集成方法？

调参