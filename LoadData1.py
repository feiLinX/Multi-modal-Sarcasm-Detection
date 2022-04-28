import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader,random_split
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import PIL
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKING_PATH=""
TEXT_LENGTH=75
TEXT_HIDDEN=256
"""
read text file, find corresponding image path
"""
def load_data():
    data_set_train=dict()
    data_set_test=dict()
    data_set_valid=dict()
    for dataset in ["train"]:
        file=open(os.path.join(WORKING_PATH,"text_data/",dataset+".txt"),"rb")
        for line in file:
            content=eval(line)
            image=content[0]
            sentence=content[1]
            group=content[2]
            if os.path.isfile(os.path.join(WORKING_PATH,"image_data/train",image+".jpg")):
                data_set_train[int(image)]={"text":sentence,"group":group}
    for dataset in ["test","valid"]:
        file=open(os.path.join(WORKING_PATH,"text_data/",dataset+".txt"),"rb")
        for line in file:
            content=eval(line)
            image=content[0]
            sentence=content[1]
            group=content[3] #2
            if os.path.isfile(os.path.join(WORKING_PATH,"image_data/test",image+".jpg")):
                data_set_test[int(image)]={"text":sentence,"group":group}
            elif os.path.isfile(os.path.join(WORKING_PATH,"image_data/valid",image+".jpg")):
                data_set_valid[int(image)]={"text":sentence,"group":group}
    return data_set_train, data_set_test, data_set_valid

data_set_train, data_set_test, data_set_valid = load_data()

def load_word_index():
    word2index=pickle.load(open(os.path.join(WORKING_PATH,"text_embedding/vocab.pickle"), 'rb'), encoding='latin1')
    return word2index
word2index=load_word_index()

def load_image_labels():
    # get labels
    img2labels=dict()
    with open(os.path.join(WORKING_PATH,"multilabel_database/","img_to_five_words.txt"),"rb") as file:
        for line in file:
            content=eval(line)
            img2labels[int(content[0])]=content[1:]
    # label to index of embedding, dict, word:value 0~1001
    label2index=pickle.load(open(os.path.join(WORKING_PATH,"multilabel_database_embedding/vocab.pickle"), 'rb'))
    return img2labels,label2index
img2labels,label2index=load_image_labels()
# save to dataloader
class my_data_set(Dataset):
    def __init__(self, data):
        self.data=data
        self.image_ids=list(data.keys())
        for id in data.keys():
          if os.path.isfile(os.path.join(WORKING_PATH,"image_data/train",str(id)+".jpg")):
            self.data[id]["image_path"] = os.path.join(WORKING_PATH,"image_data/train/",str(id)+".jpg")
          elif os.path.isfile(os.path.join(WORKING_PATH,"image_data/test",str(id)+".jpg")):
            self.data[id]["image_path"] = os.path.join(WORKING_PATH,"image_data/valid/",str(id)+".jpg")
          elif os.path.isfile(os.path.join(WORKING_PATH,"image_data/valid",str(id)+".jpg")):
            self.data[id]["image_path"] = os.path.join(WORKING_PATH,"image_data/valid/",str(id)+".jpg")

        # load all text
        for id in data.keys():
            text=self.data[id]["text"].split()
            text_index=torch.empty(TEXT_LENGTH,dtype=torch.long)
            curr_length=len(text)
            for i in range(TEXT_LENGTH):
                if i>=curr_length:
                    text_index[i]=word2index["<pad>"]
                elif text[i] in word2index:
                    text_index[i]=word2index[text[i]]
                else:
                    text_index[i]=word2index["<unk>"]
            self.data[id]["text_index"] = text_index

    # load image feature data - resnet 50 result
    
    def __image_feature_loader(self,id):
        attribute_feature = np.load(os.path.join(WORKING_PATH,"image_feature_data",str(id)+".npy"))
        return torch.from_numpy(attribute_feature)
    

    # load attribute feature data - 5 words label
    def __attribute_loader(self,id):
        labels=img2labels[id]
        label_index=list(map(lambda label:label2index[label],labels))
        return torch.tensor(label_index)

    def __text_index_loader(self,id):
        return self.data[id]["text_index"]

    def image_loader(self,id):
        path=self.data[id]["image_path"]
        img_pil =  PIL.Image.open(path)
        transform = transforms.Compose([transforms.Resize((448,448)),
                                        transforms.ToTensor(),
                                        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])
        img_tensor = transform(img_pil)
        return img_tensor
    def text_loader(self,id):
        return self.data[id]["text"]
    def label_loader(self,id):
        return img2labels[id]


    def __getitem__(self, index):
        id=self.image_ids[index]
        # img = self.__image_loader(id)
        text_index = self.__text_index_loader(id)
        image_feature = self.__image_feature_loader(id)
        attribute_index = self.__attribute_loader(id)
        group = self.data[id]["group"]
        return text_index,image_feature,attribute_index,group,id

    def __len__(self):
        return len(self.image_ids)


# train, val, test, split
train_set = my_data_set(data_set_train)
test_set = my_data_set(data_set_test)
valid_set = my_data_set(data_set_valid)

batch_size=32 #32

# add to dataloader
# all_loader = DataLoader(all_Data,batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
play_loader = DataLoader(test_set,batch_size=1, shuffle=False)

"""
example of the data
"""
if __name__ == "__main__":
    for text_index, image_feature,attribute_index,group,id in test_loader:
        # plt.imshow(img[0].permute(1,2,0))
        # plt.show()
        print("text",text_index,text_index.shape,text_index.type())
        print("image feature",image_feature.shape,image_feature.type())
        print("attribute index",attribute_index.shape,attribute_index.type())
        print("group",group,group.type())
        print("image id",id,id.type())
        break


