# SCMHR
 scalable cross modal haptic signal generation 
 ## Setup
  We run the program on a desktop using python.  
  Environment requirements: Torch 2.1.20+cu111 Torchvision 0.16.2+cu111 Tensorboard 2.12.0

## The introduction of  the code
   dataset_V_A_H.py is the code of data acquisition.  
   metric.py defines the code for different metrics.  
   new_haptic_encoder_gen.py is the different model for our SCMHR.  
   new_train_v_a_2_haptic_v3_gan.py is the code for model train.  
   utils.py is the configuration file.  
   
  ## Train the model
```
python new_train_v_a_2_haptic_v3_gan.py
```

## Test the model
  ### For the coarse-gained classification:
```
    python test/Coarse-grained classification/test_for_fenlei_coarse.py
```
 ### For the fine-gained classification:
```
  python  test/fine-grained classification task3/test_for_fenlei_fine.py
```
### For the generation task:
```
   python test/generation/test_1_model2txt.py
```
### Data
The original dataset is from LMT-108-Surface-Material database which can be downloaded at https://zeus.lmt.ei.tum.de/downloads/texture/. 
The dataset we constructed for our practical cross-modal communication platform is available at https://github.com/dadajuan/VisTouchdataset.
