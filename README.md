
<div align="center"> 

<h2> 
Single Image Dehazing Based on NDC-Net Optimized Dark Channel Prior 
</h2>
</div>


### Environment

1. Clone this repo:

```
git clone https://github.com/NTMa-IMU/NDCNet.git
```

### Training

1. Run the following script to train NDC-Net:

```
python train.py
```

### Evaluation

1. Subsequent releases of pre-trained models
2. Run the following script to get the normalized haze-free image dark channel map using the pre-trained model
```
python test.py
```
3. Run the following script for dehazing
```
python dehaze.py
```
