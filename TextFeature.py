import torch
import numpy as np
import LoadData1
from torchcrf import CRF
from AttributeFeature import ExtractAttributeFeature
# from crf import CRF
# from constants import Const
class ExtractTextFeature(torch.nn.Module):
    def __init__(self,text_length,hidden_size,dropout_rate=0.2):
        super(ExtractTextFeature, self).__init__()
        self.hidden_size=hidden_size
        self.text_length=text_length
        embedding_weight=self.getEmbedding()
        self.embedding_size=embedding_weight.shape[1]
        self.embedding=torch.nn.Embedding.from_pretrained(embedding_weight)
        self.biLSTM=torch.nn.LSTM(input_size=200,hidden_size=hidden_size,bidirectional=True,batch_first=True)
        # self.crf = CRF(hidden_size, Const.BOS_TAG_ID, Const.EOS_TAG_ID, pad_tag_id=Const.PAD_TAG_ID, batch_first=True)
        self.crf=CRF(text_length)

        # early fusion
        self.Linear_1=torch.nn.Linear(200,hidden_size)
        self.Linear_2=torch.nn.Linear(200,hidden_size)
        self.Linear_3=torch.nn.Linear(200,hidden_size)
        self.Linear_4=torch.nn.Linear(200,hidden_size)

        # dropout
        self.dropout=torch.nn.Dropout(dropout_rate)

    def forward(self, input, guidance):
      embedded=self.embedding(input).view(-1, self.text_length, self.embedding_size)
      if(guidance is not None):
        # early fusion
        # guidance is the attribute feature vector
        hidden_init=torch.stack([torch.relu(self.Linear_1(guidance)),torch.relu(self.Linear_2(guidance))],dim=0)
        cell_init=torch.stack([torch.relu(self.Linear_3(guidance)),torch.relu(self.Linear_4(guidance))],dim=0)
        output,_=self.biLSTM(embedded,(hidden_init,cell_init))
      else:
        output,_=self.biLSTM(embedded,None)

        # score, path = self.crf.decode(output, mask=mask)
        # dropout
        output=self.dropout(output)

      output=self.dropout(output)

      model_crf = self.crf
      crf_input=output.transpose(1,2)
      crf_decode=model_crf.decode(crf_input)

      RNN_state = torch.mean(output,1)
      return torch.Tensor(crf_decode).transpose(1,0), output

    def getEmbedding(self):
        return torch.from_numpy(np.loadtxt("text_embedding/vector.txt", delimiter=' ', dtype='float32'))


if __name__ == "__main__":
    test=ExtractTextFeature(LoadData1.TEXT_LENGTH, LoadData1.TEXT_HIDDEN)
    test_attr=ExtractAttributeFeature()
    for text_index,image_feature,attribute_index,group,id in LoadData1.test_loader:
        attr_result,attr_seq=test_attr(attribute_index)
        result,seq=test(text_index,attr_result)
        # [2, 512]
        print(result.shape)
        # [2, 75, 512]
        print(seq.shape)
        break
