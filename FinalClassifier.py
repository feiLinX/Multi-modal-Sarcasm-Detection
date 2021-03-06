import torch
import numpy as np
import LoadData1
import TextFeature
import AttributeFeature
import ImageFeature
import FuseAllFeature

#%%

class ClassificationLayer(torch.nn.Module):
    def __init__(self,dropout_rate=0):
        super(ClassificationLayer, self).__init__()
        self.Linear_1=torch.nn.Linear(512,256)
        self.Linear_2=torch.nn.Linear(256,2)
        self.dropout=torch.nn.Dropout(dropout_rate)
        
    def forward(self,input):
        hidden=self.Linear_1(input)
        hidden=self.dropout(hidden)
        
        output=torch.sigmoid(self.Linear_2(hidden))
        return output
        
if __name__ == "__main__":
    image=ImageFeature.ExtractImageFeature()
    text=TextFeature.ExtractTextFeature(LoadData1.TEXT_LENGTH, LoadData1.TEXT_HIDDEN)
    attribute=AttributeFeature.ExtractAttributeFeature()
    fuse=FuseAllFeature.ModalityFusion()
    final_classifier=ClassificationLayer()
    for text_index,image_feature,attribute_index,group,id in LoadData1.train_loader:
        image_result,image_seq=image(image_feature)
        attribute_result,attribute_seq=attribute(attribute_index)
        text_result,text_seq=text(text_index,attribute_result)

        output=fuse(image_result,image_seq,text_result,text_seq.permute(1,0,2),attribute_result,attribute_seq.permute(1,0,2))
        result=final_classifier(output)
        predict=torch.round(result)


        print(result.shape)
        print(result)
        print(predict)
        break