# Pytorch-3D-R<sup>2</sup>N<sup>2</sup>: 3D Recurrent Reconstruction Neural Network
This is a Pytorch implementation of the paper ["3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction"](http://arxiv.org/abs/1604.00449) by Choy et al. Given one or multiple views of an object, the network generates voxelized ( a voxel is the 3D equivalent of a pixel) reconstruction of the object in 3D.  
See [chrischoy/3D-R2N2](http://github.com/chrischoy/3D-R2N2) for the original paper author's implementation in Theano, as well as overview of the method.

## Pre-trained model
For now, only the non-residual LSTM-based architecture with neighboring recurrent unit connection is implemented. It is called *3D-LSTM-3* in the paper.  
A pre-trained model based on this architecture can be downloaded from [here](https://mega.nz/file/BHQQVJ6D#zVukPkk1dXI4qnPxzz3naoYi1RUY6wKLcLiq3q90jPU). It obtains the following result on the ShapeNet rendered images test dataset:    
IoU | Loss |
--- | --- |
0.591 | 0.093 | 

## Overview
![overview](https://github.com/Rutvikk-Khar/3D-Object-Reconstruction/assets/67324049/3d4b0db5-0b9a-43e0-be9c-8e4633d17403)
*Left: images found on Ebay, Amazon, Right: overview of `3D-R2N2`*

Traditionally, single view reconstruction and multi-view reconstruction are disjoint problems that have been dealt using different approaches. In this work, we first propose a unified framework for both single and multi-view reconstruction using a `3D Recurrent Reconstruction Neural Network` (3D-R2N2).

| 3D-Convolutional LSTM     | 3D-Convolutional GRU    | Inputs (red cells + feature) for each cell (purple) |
|:-------------------------:|:-----------------------:|:---------------------------------------------------:|
|![lstm](https://github.com/Rutvikk-Khar/3D-Object-Reconstruction/assets/67324049/c0d491de-1f59-4be7-a132-48961e7aa7b4)| ![gru](https://github.com/Rutvikk-Khar/3D-Object-Reconstruction/assets/67324049/b767bf45-3bfa-4a81-8fb6-57b324a2288f)| ![lstm_time](https://github.com/Rutvikk-Khar/3D-Object-Reconstruction/assets/67324049/9ebd230f-7301-4175-a63e-babd2d3afdcc)|

We can feed in images in random order since the network is trained to be invariant to the order. The critical component that enables the network to be invariant to the order is the `3D-Convolutional LSTM` which we first proposed in this work. The `3D-Convolutional LSTM` selectively updates parts that are visible and keeps the parts that are self-occluded.

![full_network](https://github.com/Rutvikk-Khar/3D-Object-Reconstruction/assets/67324049/f95a97ba-3c44-41e0-9f3e-27f9beebb609)
*We used two different types of networks for the experiments: a shallow network (top) and a deep residual network (bottom).*

## Results
|Input Image | Generated 3D Reconstructed Image|
|:----------:|:-------------------------------:|
|![Chair](https://github.com/Rutvikk-Khar/3D-Object-Reconstruction/assets/67324049/bcbb2f1d-3eb1-419e-8ad9-bce043b65807)| ![Screenshot 2024-05-07 215007](https://github.com/Rutvikk-Khar/3D-Object-Reconstruction/assets/67324049/9dda3c1f-2b94-46b3-905d-51cae369da1a)|
|![input_0](https://github.com/Rutvikk-Khar/3D-Object-Reconstruction/assets/67324049/faf69ed6-62d2-4b94-8793-fa77c79f876b)| ![Screenshot 2024-05-07 214316](https://github.com/Rutvikk-Khar/3D-Object-Reconstruction/assets/67324049/52f271e3-79f2-447e-93ac-a10928856ad5)|

The demo code takes 3 images of the same chair and generates the following reconstruction.
| Image 1         | Image 2         | Image 3         | Reconstruction                                                                            |
|:---------------:|:---------------:|:---------------:|:-----------------------------------------------------------------------------------------:|
| ![0](https://github.com/Rutvikk-Khar/3D-Object-Reconstruction/assets/67324049/41ff9d06-4a7e-4ee0-bc7c-8f6b1f4050c0)| ![1](https://github.com/Rutvikk-Khar/3D-Object-Reconstruction/assets/67324049/87f48b03-6d17-4b7e-b7ef-4be7d51b512a)|![2](https://github.com/Rutvikk-Khar/3D-Object-Reconstruction/assets/67324049/658b74c4-1136-4b22-b709-67e4cf12a713)|![pred](https://github.com/Rutvikk-Khar/3D-Object-Reconstruction/assets/67324049/7dce4a77-3d87-43e2-910b-cfa6b642026c)|

MeshLabs Output:
![pred](https://github.com/Rutvikk-Khar/3D-Object-Reconstruction/assets/67324049/4234cc53-1394-4796-8e79-2b1d249604b3)

## Installation
The code was tested with Python 3.6.
- Download the repository
```
git clone https://github.com/Rutvikk-Khar/3D-Object-Reconstruction
```

- Install the requirements
```
pip install -r requirements.txt
```
## Training the network

- Download and extract the ShapeNet rendered images dataset:  
```
mkdir ShapeNet/
wget http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
wget http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz
tar -xzf ShapeNetRendering.tgz -C ShapeNet/
tar -xzf ShapeNetVox32.tgz -C ShapeNet/
```

- Rename the ```config.ini.example``` config template file to e.g ```your_config.ini```, and change parameters in it as required.
- Run ```python train.py --cfg=your_config.ini```. Or simply ```python train.py``` if you named your config file ```config.ini```.

## Test your trained model
- Run ```python test.py --cfg=your_config.ini```. Or simply ```python test.py``` if your config file is named ```config.ini```.  
  This can be the same config file used for training the model. Note that when testing, you probably want to set ```resume_epoch``` to the number of epochs that your model was trained for.




