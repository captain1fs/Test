### Memory Efficient Combination of Tensor Decomposition and Recomputation for Training DNNs

#### 摘要

如今，深度神经网络的训练有着很高的GPU 内存需求，尤其是对于大规模深度神经网络。但高带宽的GPU内存是一种稀缺资源，这一限制阻碍了研究者去探索更先进的模型架构，进而阻碍了深度神经网络的进一步发展。因此，对GPU内存资源的优化利用提出了很高的要求。

面临的挑战：

本文的解决方法：张量分解压缩参数 + 重新计算，减少显存占用

所做工作：分析张量分解、重新计算，自适应结合两者对网络模型进行优化

#### 1.引言

**P1：背景**

过去十几年来，深度神经网络 (DNN) 在许多领域都取得了显著的成功，比如图像分类 [1, 2]、目标检测 [3, 4]、文本分类 [5, 6] 和自然语言处理[7,8]。越来愈多的研究 [11] 表明，更深更宽的模型可以带来更高的精度，这促使研究者们去追求更大、更具表现力的模型。如今，越来越多的研究者和研究机构都使用GPU来训练模型。然而，现有的GPU内存无法满足这些大规模模型训练[8,9,10]时的内存需求。*NVIDIA最新GPU内存只到32GB...* 此外，深度神经网络的规模呈指数级增长，而GPU内存的增长无法跟上，限制了大规模深度神经网络的进一步发展。 *BERT、GPT参数量...*

**P2：现有的工作有着缺陷或者说并不完美。**

显存优化是深度学习中一个现实的问题。

CPU-GPU交换技术：

张量分解压缩神经网络 [9,10] 是。。。

重新计算 [...] 。。。

**P3：我们的想法：**

张量分解压缩神经网络在加速模型训练的同时会降低模型大小，但会损失一些精度，这可以通过微调来弥补。重新计算能够降低模型训练期间的显存占用，但是会延长训练时间。因此将两种方法有效结合使用，可做到既降低了模型训练时的显存占用，又不会显著延长训练时间。 ...

**P4：我们的工作**

在本文中，我们提出了 CTDR，这是一种基于张量的 GPU 内存管理模块，用于深度学习框架，通过张量分解压缩神经网络和重新计算来减少内存占用。



#### 2.相关工作

**P1：现有的方法：交换、检查点、压缩**

已经提出了许多方法来减少 DNN 训练的 GPU 内存占用。然而，这些解决方案有其局限性。

**重新生成**：

有两种主要技术：交换和重新计算。这两种方法都是基于在前向传播中释放特征图的内存并在后向传播中重新生成中间特征图的设计原则。它们在如何执行再生方面有所不同。具体来说，交换利用  CPU 内存作为更大的外部存储器，并在 GPU 和 CPU 之间来回异步复制数据；而重新计算通过重复部分前向计算过程来获得所需的中间特征图。这两种方法都不会影响训练的准确性。

vDNN[12]针对优化Conv层的feature map

MoDNN[14]针对vDNN优化，核心发现是DNN框架中，卷积函数包括很多不同的类别，有的卷积函数空间使用多，但是性能较快（FFT、Winograd algorithm），有的不占用太多内存，但是性能相对较差（GEMM）。所以moDNN能够智能的选择卷积函数，并调整mini-batchsize来最优化系统的执行性能，并智能选择转移策略。

SuperNeurons[15]首次结合交换和重新计算，缺点是针对层

Capuchin[19]：可看作是对SuperNeurons的改进，以张量为粒度

SwapAdvisor[18]认为之前的基于人工判断的转移方法并不高效（例如vDNN的只对卷积转移），所以使用传统的启发式搜索方案来进行转移策略的搜索，这里选择的是遗传算法。

FalshNeurons[20]认为上述一堆文章都是把数据转移到CPU的DRAM上，但是那些文章都没有考虑过内存、CPU正在执行数据预处理操作，从而使得内存总线始终在忙碌，从而使得转移性能极差。于是另辟蹊径，将数据转移到SSD上。

ZeRO-Offload[21]针对NLP等模型，把优化器所有参数及其计算卸载到CPU，并设计了更快的CPU优化器运算，从而做到了完美的GPU显存优化。

checkpoint[13]可释放大量显存，但计算开销加剧

DTR



**压缩**：

Gist[16]针对ReLU的输出进行有损+无损的压缩，释放了许多ReLU层相关的显存。而缺点也是针对性太强，即必须对含有ReLU的模型进行操作，所以也限制了其贡献。

cDMA[17]利用了ReLU输出层的稀疏特性，并在GPU中做了一个硬件core来压缩这些数据，以减少转移的数据量，最终提升性能。但这是基于模拟器所做，也无法为我们所用，并且压缩算法定制到硬件上，也比较单一。

张量分解：压缩模型[9,10] 是。。。

CP[9]

Tucker[10]



#### 3.前提

张量分解压缩模型的理论分析



#### 4.设计方法

首先选择合适的Tensor decomposition方法来分解卷积层和全连接层的参数矩阵，具体来讲，把一个参数tensor分解成几个小的tensor，再以这些tensor建立新的卷积层。因此新的网络模型层数会增加。由于张量分解，模型参数数量会减少，计算量也会随之减少，模型的大小也会降低。代价是损失一部分精度，但可以通过增加迭代次数进行微调恢复。

然后，采用重新计算，选择某些层的输出保留，其余的全部在前向传播中抛弃，后向传播时再计算回来。之所以能够计算回来，是因为某一层虽然输出被抛弃，但输入和参数矩阵会被记录下来。由于整个过程相当于进行了两次前向传播+一次后向传播，训练时间会延长大约15%~20%左右。

**具体实现**：。。。

算法：



#### 5.实验评估

P1：实验环境：M40。。。

​		设计：首先比较优化前后模型训练时的GPU内存占用，其次比较能够达到的batch-size大小、训练时间，还要考虑精度的损失、微调的时间等等

P2：针对线性网络如 Alexnet、Lenet、VGG

P3：针对非线性网络如 Resnet、Densenet、GoogleNet等

P3：针对基于transform的网络模型

不同类型的模型优化效果可能何会不同，分析原因得出结论



#### 6.结论

做了哪些优化，减少了模型训练时的显存占用。。。。





#### 参考文献

[1] A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam, “Mobilenets: Efficient convolutional neural net-works for mobile vision applications,” arXiv preprint arXiv:1704.04861, 2017.
[2] X. Zhang, X. Zhou, M. Lin, and J. Sun,“Shufflenet: An extremely efficient convolutional neural network for mobile devices,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 6848–6856.
[3] R. Girshick,“Fast r-cnn,” in Proceedings of the IEEE in-ternational conference on computer vision, 2015, pp. 1440–
1448.
[4] S. Ren, K. He, R. Girshick, and J. Sun,“Faster r-cnn: To-wards real-time object detection with region proposal networks,” Advances in neural information processing sys-tems, vol. 28, pp. 91–99, 2015.

[5] S. Hochreiter and J. Schmidhuber,“Long short-termmemory,” Neural computation, vol. 9, no. 8, pp. 1735–1780, 1997.
[6] J. Zhou, C. Ma, D. Long, G. Xu, N. Ding, H. Zhang, P. Xie, and G. Liu,“Hierarchy-aware global model for hierarchical text classification,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2020, pp. 1106–1117.
[7] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in neural infor-mation processing systems, 2017, pp. 5998–6008.
[8] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transform-ers for language understanding,” arXiv preprint arXiv:1810.04805, 2018.

[9] V. Lebedev, Y. Ganin, M. Rakhuba, I. Oseledets, V. Lempitsky. Speeding-up convolutional neural networks using fine-tuned CP-decomposition. https://arxiv.org/pdf/1412.653.pdf, 2015.

[10] Kim, Yong-Deok & Park, Eunhyeok & Yoo, Sungjoo & Choi, Taelim & Yang, Lu & Shin, Dongjun. (2016). Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications. 

[11] Soheil Bahrampour, Naveen Ramakrishnan, Lukas Schott, and Mohak Shah. 2015. Comparative study of deep learning software frameworks. arXiv:1511.06435 (2015).

[12] M. Rhu, N. Gimelshein, J. Clemons, A. Zulfiqar, and S. W. Keckler, “**VDNN**: Virtualized deep neural networks for scalable, memory-efficient neural network design,” in Proceedings of the Annual International Symposium on Microarchitecture, MICRO, 2016, vol. 2016-Decem.

[13] T. Chen, B. Xu, C. Zhang, and C. Guestrin, “Training Deep Nets with Sublinear Memory Cost,” pp. 1–12, 2016.

[14] X. Chen, D. Z. Chen, and X. S. Hu, “**MoDNN**: Memory optimal DNN training on GPUs,” Proc. 2018 Des. Autom. Test Eur. Conf. Exhib. DATE 2018, vol. 2018-Janua, pp. 13–18, 2018.

[15] L. Wang et al., “**SuperNeurons**: Dynamic GPU memory management for training deep neural networks,” in Proceedings of the ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming, PPOPP, 2018, pp. 41–53.

[16] A. Jain, A. Phanishayee, J. Mars, L. Tang, and G. Pekhimenko, “**GIST**: Efficient data encoding for deep neural network training,” Proc. - Int. Symp. Comput. Archit., pp. 776–789, 2018.

[17] M. Rhu, M. O’Connor, N. Chatterjee, J. Pool, Y. Kwon, and S. W. Keckler, “**Compressing DMA Engine**: Leveraging Activation Sparsity for Training Deep Neural Networks,” Proc. - Int. Symp. High-Performance Comput. Archit., vol. 2018-Febru, pp. 78–91, 2018.

[18] C. C. Huang, G. Jin, and J. Li, “**SwapAdvisor**: Pushing deep learning beyond the GPU memory limit via smart swapping,” in International Conference on Architectural Support for Programming Languages and Operating Systems - ASPLOS, 2020, pp. 1341–1355.

[19] X. Peng et al., “**Capuchin**: Tensor-based GPU memory management for deep learning,” in International Conference on Architectural Support for Programming Languages and Operating Systems - ASPLOS, 2020, pp. 891–905.

[20] J. Bae et al., “**FlashNeuron** : SSD-Enabled Large-Batch Training of Very Deep Neural Networks This paper is included in the Proceedings of the 19th USENIX Conference on File and Storage Technologies .,” 2021.

[21] J. Ren et al., “**ZeRO-Offload** : Democratizing Billion-Scale Model Training,” 2021.