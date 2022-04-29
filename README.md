# pytorch-multimodal_sarcasm_detection
It is the pytorch implementation of paper "Multi-Modal Sarcasm Detection in Twitter with Hierarchical Fusion Model"
# Overview
![20201108153958](https://user-images.githubusercontent.com/7517810/98483722-b9df3e00-21d8-11eb-9ece-fb05e265bcf5.png)
# Data and original implementation
The image data and original implementaion(Tensorflow v1) can be found from [here](https://github.com/headacheboy/data-of-multimodal-sarcasm-detection)
## References
1.  **Multi-Modal Sarcasm Detection in Twitter with Hierarchical Fusion Model**<br />
    Yitao Cai, Huiyu Cai and Xiaojun Wan. <br />
    [[link]](https://www.aclweb.org/anthology/P19-1239/). In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (pp. 2506-2515).(2019)

# What pys for:
1. ImageFeatureDataGenerator.py: to obatain raw image vector.

2. ImageFeature.py: to obtain image guidance vector. Note that the method they used was kind of different from
what's been talked in the paper, where they let those raw image vector passed through a one-layer NN.

3. AttributeFeature.py: to obtain raw attribute vector and attribute guidance vector. Note that they directly used the 
word embeddings of the five predicted attributes of those images, and they didn't provide code showing how they 
obtain them, like saying in the paper (ResNet-101......)

4. TextFeature.py: to obtain raw text feature (hidden states in each time step) and text guidance vector, and applied
early fusion. They also directly used the word embedding of the text and didn't provide code showing how they
generate the word embeddings saying in the paper (GloVe.....)

5. FuseAllFeature.py: apply representation fusion and modality fusion

6. FinalClassifier.py: the final classification step: a 2-layer NN.

7. LoadData.py: create dataloader to load text, image, and attribute data

# How to run:
First, enter "sarcacm_run.ipynb", follow the cells to download image data and image feature data, and place them into right order. There should be 24635 .npy files for image feature. Then enter "run.ipynb" for model implementation and testing. You may just ignore "Colab: Connect Google Drive" part if you're not going to use Colab, and also play with "Test" part to check if all .py work well (you may ignore this part as well). Then run "Train & Test" part.

# You may also use links below to download image data and image feature data:
Image data: https://github.com/headacheboy/data-of-multimodal-sarcasm-detection

image feature data: https://drive.google.com/drive/folders/1scn5tk8LObL4VAzE6j5KHj5ESXMvAWNH?usp=sharing

Trained BERTWeet model: https://drive.google.com/file/d/1O6taOZ6plbT8FGg-5hufbkpMxZiz-2Hj/view?usp=sharing
