This project enables the use of a conditional generative adversarial network (CGAN)
and data scraping of coin databases. The intent is to reconstruct partial or incomplete 
coins using a CGAN.

# Coin Reconstruction using CGAN

The use of Generative Adversarial Networks (GANs) for reconstruction of archaeological objects is a sparsely researched field. Ancient objects are often found broken and/or missing significant parts. As an example, ancient coins are often found warn or broken in places, particularly after being buried for centuries. This can make the classification and reconstruction of such objects challenging. A methodology that can reconstruct ancient cultural items or objects could potentially be useful for archaeologists and other specialists in enabling the classification of even less preserved or damaged objects.



## Installation
This code can ruu on Google Collab or through Command Line  
- The following link is for Google Collab 
 https://colab.research.google.com/drive/11t44mEcfBRKvQxpC4V6hafia52WfiDtD?usp=sharing

- Have Python installed:

Python : https://www.python.org/downloads/

- PIP Installation:

```bash
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
```
- Reopen the command line prompt and make sure that python is installed :
```bash
Python --version
```
- Download Anaconda before downloading PyTorch. Anaconda would take a bit of time to download.
Installation of all imports:
1.	Torch: 
```bash
pip install torch
```
 
2.	Torchvision :
```bash
 pip install torchvision
``` 
3.	CV2: 
```bash
pip install opencv-python
```
4.	Pandas:  
```bash
pip install pandas
```
5.	Scipy : 
```bash
pip install scipy
``` 
6.	Matplotlib: 
```bash
pip install matplotlib
``` 



    
## Features

- Dataset uploading
- Preparation of Raw Dataset to Dataframe
- Display of Dataset images for validation 
- Data distribution visualization
- Train - Test Split 
- Training-set distribution balancing
- Dataset statistics
- Preprocessing of Data
- CGAN for reconstruction
- Training of Network using specified parameters
- Network Evaluation


## Authors

- [@adelkhelifi](https://www.github.com/octokatherine)
- [@markaltaweel](https://github.com/maltaweel)
- [@hurmaehtesham](https://github.com/hurmaeht)


