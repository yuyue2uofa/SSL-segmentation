# SSL-segmentation
# Self-supervised learning to more efficiently generate segmentation masks for wrist ultrasound
This is the code for MICCAI paper: [Self-supervised learning to more efficiently generate segmentation masks for wrist ultrasound](https://link.springer.com/chapter/10.1007/978-3-031-44521-7_8)

The TransUNet folder files are from TransUNet github(by Chen et al.) with bugs fixed and model added a sigmoid activation function: https://github.com/Beckschen/TransUNet/tree/main/networks. 

To download TransUNet pretrained on ImageNet, provided by TransUNet authors Chen et al.(R50+ViT-B_16.npz): https://console.cloud.google.com/storage/browser/vit_models/imagenet21k?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false

To pretrain model using modified SimMIM: python SimMIM_pretrain.py

To visualize SSL pretrained model reconstruction: python SimMIM_visualize.py

To finetune model for image segmentation: python segmentation_model_finetune.py

To determine segmentation threshold based on validation set using Otsu’s method: python segmentataion_threshold.py

To evaluate model performance and save segmentation prediction on test set: python segmentation_model_evaluation.py

Modified SimMIM for TransUNet and UNet: SimMIM_model


