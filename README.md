# Transfer Learning for Music Genre Classification

# Overview

# Prerequisites and Dependencies
You will need the following packages to run the code. To install the packages used in this project, run the following command.
```
$ pip install jupyter
$ pip install pandas
$ pip install numpy
$ pip install scipy
$ pip install tensorflow==1.4.0
$ pip install keras==2.1.2
$ pip install librosa
$ git clone https://github.com/keunwoochoi/kapre.git
$ cd kapre
$ python setup.py install
```
Our code is based on python version 3.6

If you are using Anaconda, install *ffmpeg* by calling
```
conda install -c conda-forge ffmpeg
```

If you are not using Anaconda, here are some common commands for different operating systems:

* Linux (apt-get): `apt-get install ffmpeg` or `apt-get install gstreamer1.0-plugins-base gstreamer1.0-plugins-ugly`
* Linux (yum): `yum install ffmpeg` or `yum install gstreamer1.0-plugins-base gstreamer1.0-plugins-ugly`
* Mac: `brew install ffmpeg` or `brew install gstreamer`
* Windows: download binaries from the website

# Running the code

To test and run this code, you will need the following:
  * data.txt  //TODO link data.txt
  * corresponding images in .png format that matches the `data.txt` //TODO link images 
  * svm.py


# Credit 

Pre-trained Convnet: https://github.com/keunwoochoi/transfer_learning_music

FMA dataset: https://github.com/mdeff/fma

* Please let me know if I fortget any citation, credit or made any mistake. :)