~~ImageNet106Mins~~

# ImageNet104Mins

<b>Updates</b>:
Imagenet Training to reach 93% in Top 5 Acc in 6273.32 sec (step 15000, 104Mins)

~~Imagenet Training in 106 Mins to reach 93% in Top 5 Acc~~

The experiments were ran on Amazon AWS Sagemaker, and part of resnet50 model code was from AWS (https://github.com/aws-samples/deep-learning-models/tree/master/models/resnet/tensorflow)

To use the horovod in sagemaker, training and evaluation have to be separated, so we train the imagenet for 40 epochs and then evaluate those 40 models asynchronously on 8 gpus (each GPU evaluates 5 models on full dataset)

<b>Note:</b> we don't use fastai's rectangular evaluation on the valiation data (acutually, we did the experiment and found this may not help when we use the tensorflow), to achieve the 93%, we use the traditional single center cropping.

The experiments are reproducible in Sagemaker:

Training Experiment ID: 

~~https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs/imagenet-ml-p3-16xlarge-2018-12-30-03-23-22-758~~

https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs/imagenet-ml-p3-16xlarge-2019-01-04-22-50-34-217-copy-2

Replicated Experiment:

https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs/imagenet-ml-p3-16xlarge-2018-12-30-03-23-22-758-copy-12-30

Both Experiments reached 93% in TOP5 Acc by steps 15625, around 106 Mins, filter the cloudwatch log by typing "Log".

Model Artifacts are also available by downloading the models through the sagemaker experiment

Strategies used to achieved the goal:

- Distributed Multi-GPU Computing (Horovod)
- Multi-res image training strategy (Low resolution -> High resolution)
- Dynamic Batching strategy
- Dynamic Learning rate scheduler
- Mixed Precision Training (FP32+FP16)

Training Strategy:
```
training_strategy = [
    {'epoch':[0,6], 'lr': [1.0,2.0],'lr_method':'linear','batch_size':740, 'image_size':(128, 128), 'data_dir':'160', 'prefix':'train'},
    {'epoch':[6,21], 'lr': [2.0,0.45],'lr_method':'linear','batch_size':740, 'image_size':(128, 128), 'data_dir':'160', 'prefix':'train'},
    {'epoch':[21,32], 'lr': [0.45,0.02],'lr_method':'exp','batch_size':256, 'image_size':(224, 224), 'data_dir':'320', 'prefix':'train'},
    {'epoch':[32,36], 'lr': [0.02,0.004],'lr_method':'exp','batch_size':196, 'image_size':(224, 224), 'data_dir':'320', 'prefix':'train'},
    {'epoch':[36,40], 'lr': [0.004,0.002],'lr_method':'exp','batch_size':128, 'image_size':(288, 288), 'data_dir':'320', 'prefix':'train'}
    ]
```

Image Data (resized image dataset from fastai): 
- https://s3.amazonaws.com/yaroslavvb/imagenet-sz.tar 
- You need to use build_imagenet_data.py to build the tf records from those resized images

