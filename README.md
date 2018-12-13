# ece285_mlip

## Report
https://www.overleaf.com/2891186856zbwqrpsxxmgd

## Download the Original Pix3D Dataset  
> download_dataset.sh

## Render the Synthetic Image
Install Belender first.  
> https://www.blender.org/download/  

Run the "blender_renderer.ipynb"

## Segementing the foreground and background Images
Run the "background_segementation.ipynb"

## Demo
Run the "Demo.ipynb"
The pretrained weight is stored in "Demo_Generator.pt"

## Train the network
The deafult configuration is using NLayer discriminator and MSE Loss.  
To train the network, simply use
> python net.py


To change the training epochs and batch size, try
> python net.py --epochs 100 --batchSize 32  
> python net.py -e 100 --batchSize 32  

To change the learning rate, try  
> python net.py --lr_G 0.01 --lr_D 0.01  

To use the pixel discriminator, use  
> python net.py --D_net Pix  

To use binary cross entropy loss, use  
> python net.py --GAN_Loss BCE  

