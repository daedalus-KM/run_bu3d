# run_bu3d
- How to run

baed on OS: Linux, GPU: A100, python = 3.10 

- install labraries

  
  conda install -y -c conda-forge cudatoolkit==11.8

  conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

  conda install numpy, matplotlib, scipy, jax, h5py, tensorboard, mat73

- git clone
  
  git clone https://github.com/daedalus-KM/run_bu3d.git

- move directory

  cd run_bu3d 

  cd bu3d

  cd gen_data

- download middlebury dataset to create histogram
  
  bash download_data.sh 0

- create histogram, and construct initial mutliscae depth maps
  
  python create_dataset_middlebury.py

- move directory
  
  cd..  

  cd train 

- Train BU3D model
  
  python train.py 

- move directory
  
  cd ..

  cd run

- Inference trained model
  
  python test_middlebury1024
