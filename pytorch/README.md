# Deep Transfer Learning on PyTorch

This is a PyTorch library for deep transfer learning. ~~We use the PyTorch version 0.2.0\_3.~~  We use the pytorch version 0.4.0.



#### Main Updates

* Change Pytorch to version 0.4.0
* Adapt to Python 3.6
* Train on Windows and new CUDA

#### ToDo

* Adapt to Python 3.6
* Adapt to new version of pytorch



## Prerequisites
~~Linux or OSX~~

~~NVIDIA GPU + CUDA-7.5 or CUDA-8.0 and corresponding CuDNN~~

~~PyTorch~~

~~Python 2.7 (We have not test on Python 3 yet.)~~

We test it on Windows 10

NVIDIA GPU 1060 + CUDA 9.1 + CuDNN 7.1

Pytorch 0.4.0

Python 3.6

## Data Preparation
---------------
In `data/office/*.txt`, we give the lists of three domains in [Office](https://cs.stanford.edu/~jhoffman/domainadapt/#datasets_code) dataset. 

Data download from: [https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view)

## Training Model
---------------
In `src` directory, you can use the following command to train the model.
```shell
python train.py gpu_id
```

To train your model with different methods or different datasets or different optimizers, you can construct your own configuration. We have given a example configuration in `train.py`. We will give some explanation.

`config['loss']` is the loss configuration, you need to set the `name` parameter as the name of the loss you want to use `DAN`, `RTN` or `JAN`. You also need to set the `trade_off` parameter to set the trade-off between the classification loss and transfer loss. If you'd like to use different parameters for the specific loss, you can set `params`, which is a dictionary including the parameters of the specific loss. The parameters of each specific loss is in `loss.py`.

`config['data']` set the input dataset parameters. You can change the dataset to your dataset here.

## Citation
If you use this library for your research, we would be pleased if you cite the following papers:

```latex
    @inproceedings{DBLP:conf/icml/LongC0J15,
      author    = {Mingsheng Long and
                   Yue Cao and
                   Jianmin Wang and
                   Michael I. Jordan},
      title     = {Learning Transferable Features with Deep Adaptation Networks},
      booktitle = {Proceedings of the 32nd International Conference on Machine Learning,
                   {ICML} 2015, Lille, France, 6-11 July 2015},
      pages     = {97--105},
      year      = {2015},
      crossref  = {DBLP:conf/icml/2015},
      url       = {http://jmlr.org/proceedings/papers/v37/long15.html},
      timestamp = {Tue, 12 Jul 2016 21:51:15 +0200},
      biburl    = {http://dblp2.uni-trier.de/rec/bib/conf/icml/LongC0J15},
      bibsource = {dblp computer science bibliography, http://dblp.org}
    }
    
    @inproceedings{DBLP:conf/nips/LongZ0J16,
      author    = {Mingsheng Long and
                   Han Zhu and
                   Jianmin Wang and
                   Michael I. Jordan},
      title     = {Unsupervised Domain Adaptation with Residual Transfer Networks},
      booktitle = {Advances in Neural Information Processing Systems 29: Annual Conference
                   on Neural Information Processing Systems 2016, December 5-10, 2016,
                   Barcelona, Spain},
      pages     = {136--144},
      year      = {2016},
      crossref  = {DBLP:conf/nips/2016},
      url       = {http://papers.nips.cc/paper/6110-unsupervised-domain-adaptation-with-residual-transfer-networks},
      timestamp = {Fri, 16 Dec 2016 19:45:58 +0100},
      biburl    = {http://dblp.uni-trier.de/rec/bib/conf/nips/LongZ0J16},
      bibsource = {dblp computer science bibliography, http://dblp.org}
    }
    
    @inproceedings{DBLP:conf/icml/LongZ0J17,
      author    = {Mingsheng Long and
                   Han Zhu and
                   Jianmin Wang and
                   Michael I. Jordan},
      title     = {Deep Transfer Learning with Joint Adaptation Networks},
      booktitle = {Proceedings of the 34th International Conference on Machine Learning,
               {ICML} 2017, Sydney, NSW, Australia, 6-11 August 2017},
      pages     = {2208--2217},
      year      = {2017},
      crossref  = {DBLP:conf/icml/2017},
      url       = {http://proceedings.mlr.press/v70/long17a.html},
      timestamp = {Tue, 25 Jul 2017 17:27:57 +0200},
      biburl    = {http://dblp.uni-trier.de/rec/bib/conf/icml/LongZ0J17},
      bibsource = {dblp computer science bibliography, http://dblp.org}
    }
```

## Contact
If you have any problem about this library, please create an Issue or send us an Email at:
- caozhangjie14@gmail.com
- longmingsheng@gmail.com
