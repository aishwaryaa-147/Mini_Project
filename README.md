# ScaDLES: {Sca}lable {D}eep {L}earning over {S}treaming data at the {E}dge

**Implementation and data-streams simulation for the work _ScaDLES_ presented at IEEE International Conference on Big Data, 2022, Osaka, Japan.**

_Distributed deep learning (DDL) training systems are designed for cloud and data-center environments that as- sumes homogeneous compute resources, high network bandwidth, sufficient memory and storage, as well as independent and identically distributed (IID) data across all nodes. 
However, these assumptions don’t necessarily apply on the edge, especially when training neural networks on streaming data in an online manner. 
Computing on the edge suffers from both systems and statistical heterogeneity. 
Systems heterogeneity is attributed to differences in compute resources and bandwidth specific to each device, while statistical heterogeneity comes from unbalanced and skewed data on the edge. 
Different streaming-rates among devices can be another source of heterogeneity when dealing with streaming data. 
If the streaming rate is lower than training batch-size, device needs to wait until enough samples have streamed in before performing a single iteration of stochastic gradient descent (SGD). 
Thus, low-volume streams act like stragglers slowing down devices with high-volume streams in synchronous training. 
On the other hand, data can accumulate quickly in the buffer if the streaming rate is too high and the devices can’t train at line-rate. 
In this paper, we introduce ScaDLES to efficiently train on streaming data at the edge in an online fashion, while also addressing the challenges of limited bandwidth and training with non-IID data. 
We empirically show that ScaDLES converges up to 3.29× faster compared to conventional distributed SGD._

**ACCESS LINKS**
- [Link1](https://ieeexplore.ieee.org/document/10020597)
- [Link2](https://sahiltyagi.academicwebsite.com/publications/21209-scadles-scalable-deep-learning-over-streaming-data-at-the-edge)

**RUNNING**

- For streaming simulation, we need Apache Zookeeper and Apache Kafka (tested with [v3.1.1](https://archive.apache.org/dist/kafka/3.1.1/RELEASE_NOTES.html)) services.
- Communication backend used currently is ```MPI```. Can be changed to ```Gloo``` or ```NCCL``` (for training on NVIDIA GPUs) by changing the ```_backend_``` parameter in the execution scripts.
- Assuming the repository is cloned into the ```home``` directory, scripts to run can be found in ```execution_scripts``` directory.
- To train models by simulating streaming-data, run ```streamsim_resnet152.sh``` or ```streamsim_vgg19.sh```.
- To train on static compression, run ```staticcompression_resnet152.sh``` or ```staticcompression_vgg19.sh```.
- For randomized data-injection training, run ```datainjection_resnet152.sh``` or ```datainjection_vgg19.sh```.
- For adaptive compression based on gradient-sensitivity and critical regions, run ```adapcomp_resnet152.sh``` or ```adapcomp_vgg19.sh```.
- For adaptive compression with randomized data-injection over streaming-data at varying stream-rates, run ```adapcomp_injection_resnet152.sh``` or ```adapcomp_injection_vgg19.sh```.

**CITATION**
- **_Bibtex_**: @article{Tyagi2022ScaDLESSD,
                 title={ScaDLES: Scalable Deep Learning over Streaming data at the Edge},
                 author={Sahil Tyagi and Martin Swany},
                 journal={2022 IEEE International Conference on Big Data (Big Data)},
                 year={2022},
                 pages={2113-2122}}