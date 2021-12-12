import sys
#import tflearn
import time
import pickle
import random

import pickle
import numpy as np
#patch()
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use("QtAgg")

import torch
import pandas as pd
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from imblearn.under_sampling import RandomUnderSampler

from sklearn import metrics

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
data = None 
with open('btc_past_1m/data.pkl', 'rb') as handle:
    data = pickle.load(handle)

length  = len(data['data'])
X_train = data['data'][0:int(length*0.7)]
y_train = data['label'][0:int(length*0.7)]
idx_train = [[i] for i,x in enumerate(X_train)]
undersample = RandomUnderSampler(sampling_strategy='majority')
idx_train,y_train = undersample.fit_resample(idx_train, y_train)
#idx_train = [x[0] for x in idx_train]
X_train = [X_train[idx[0]] for idx in idx_train]

X_test = data['data'][int(length*0.7)+1:]
y_test = data['label'][int(length*0.7)+1:]
#print(device)




classes = ('Neutral profit','Neutral Loss', 'Loss','Profit')
class BinanceDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X,y,transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.X = X
        self.y = y
        self.transform = transform
        

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #landmarks = self.landmarks_frame.iloc[idx, 1:]
        data = np.array(self.X[idx])
        data = data.astype('float32')#.reshape(-1, 2)
        max_ = max(map(max, data))
        data = 2*data/max_ - 1
        #print(data)
        label =  self.y[idx]
        sample = {'data': data, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

#transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = BinanceDataset(X_train,y_train)
test_dataset = BinanceDataset(X_test,y_test)
train_data = DataLoader(train_dataset, batch_size=32,
                        shuffle=True, num_workers=1)

test_data = DataLoader(test_dataset, batch_size=32,
                        shuffle=False, num_workers=1)
print(min(map(min,train_dataset[2]['data'])))
#while(True): continue;

class SimpleModel(nn.Module):
    def __init__(self, input_size,seq_len,output_size):
        super(SimpleModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.bn0 = nn.BatchNorm1d(self.seq_len*self.input_size)
        self.bn1 = nn.BatchNorm1d(1024) 
        self.fc1 = nn.Linear(self.seq_len*self.input_size,1024)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(1024,128)
        self.fc3 = nn.Linear(128,self.output_size)

    def forward(self, x):
        x = torch.reshape(x, (-1,self.seq_len*self.input_size))
        x = self.bn0(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        #print(x.shape)
        return x


class Seq2SeqAttention(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, hidden_layers = 2,  bidirectional = True):
                super(Seq2SeqAttention, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.output_size = output_size
                self.bidirectional = bidirectional
                self.hidden_layers = hidden_layers

                #self.config = config

                # Embedding Layer
                #self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
                #self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)

                # Encoder RNN
                self.lstm = nn.GRU(input_size = self.input_size,
                                                        hidden_size = self.hidden_size,
                                                        num_layers = self.hidden_layers,
                                                        bidirectional = self.bidirectional)

                # Dropout Layer
                #self.dropout = nn.Dropout(0.1)

                # Fully-Connected Layer
                self.fc = nn.Linear(
                        self.hidden_size * (1+self.bidirectional) * 2,
                        32
                )

                self.fc1 = nn.Linear(
                        32,self.output_size
                )

                # Softmax non-linearity
                #self.softmax = nn.Softmax()

        def apply_attention(self, rnn_output, final_hidden_state):
                '''
                Apply Attention on RNN output
                
                Input:
                        rnn_output (batch_size, seq_len, num_directions * hidden_size): tensor representing hidden state for every word in the sentence
                        final_hidden_state (batch_size, num_directions * hidden_size): final hidden state of the RNN
                        
                Returns:
                        attention_output(batch_size, num_directions * hidden_size): attention output vector for the batch
                '''
                hidden_state = final_hidden_state.unsqueeze(2)
                attention_scores = torch.bmm(rnn_output, hidden_state).squeeze(2)
                soft_attention_weights = F.softmax(attention_scores, 1).unsqueeze(2) #shape = (batch_size, seq_len, 1)
                attention_output = torch.bmm(rnn_output.permute(0,2,1), soft_attention_weights).squeeze(2)
                return attention_output


        def forward(self, x):
                # x.shape = (max_sen_len, batch_size)
                # embedded_sent = self.embeddings(x)
                # embedded_sent.shape = (max_sen_len=20, batch_size=64,embed_size=300)

                ##################################### Encoder #######################################
                lstm_output, h_n = self.lstm(x)
                # lstm_output.shape = (seq_len, batch_size, num_directions * hidden_size)

                # Final hidden state of last layer (num_directions, batch_size, hidden_size)
                #print(h_n.shape)
                batch_size = h_n.shape[1]
                h_n_final_layer = h_n.view(self.hidden_layers,
                                                                   self.bidirectional + 1,
                                                                   batch_size,
                                                                   self.hidden_size)[-1,:,:,:]

                ##################################### Attention #####################################
                # Convert input to (batch_size, num_directions * hidden_size) for attention
                final_hidden_state = torch.cat([h_n_final_layer[i,:,:] for i in range(h_n_final_layer.shape[0])], dim=1)

                attention_out = self.apply_attention(lstm_output.permute(1,0,2), final_hidden_state)
                # Attention_out.shape = (batch_size, num_directions * hidden_size)

                #################################### Linear #########################################
                concatenated_vector = torch.cat([final_hidden_state, attention_out], dim=1)
                #final_feature_map = self.dropout(concatenated_vector) # shape=(batch_size, num_directions * hidden_size)
                first_fc = self.fc(concatenated_vector)
                #print(first_fc.shape)
                relu_out = F.relu(first_fc)
                final_out = self.fc1(relu_out)
                return final_out

def test_func(net,test_data):
    global classes
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    TP = 0.0
    FP = 0.0
    FN = 0.0
    scores = []
    GT  = []
    with torch.no_grad():
        for data in test_data:
            #images, labels = data
            inputs = data['data'].to(device)
            labels = data['label'].to(device)
            outputs = net(inputs)
            probs =  F.softmax(outputs)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction,prob in zip(labels, predictions,probs):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                    if label == 3:
                        TP+=1
                elif prediction == 3:
                    FP+=1
                elif label==3:
                    FN+=1
                scores.append(prob[prediction].item())
                GT.append(label.item())
                total_pred[classes[label]] += 1
    
    
    # print accuracy for each class
    precision = TP/(FP+TP)*100
    #hit_rate = TP/(FN+)    
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                       accuracy))
    #print("Precision: ",precision)
    #fpr, tpr, thresholds = metrics.roc_curve(GT, scores, pos_label=3)
    #print("FPR: ",fpr)
    #print("FPR: ",tpr)
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    prec, recall, _ = precision_recall_curve(GT, scores, pos_label=3)
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
    plt.show()
    #for p,r in zip(prec,recall):
    #    print(p,r)
#net = Seq2SeqAttention(5,12, 2)
net = SimpleModel(5,300,4)
##net = torch.load("../../../../models_attention_2/model_99.pt")
net = net.to(device)
dataiter = iter(train_data)
data_t = dataiter.next()['data'].to(device)
#data_t = data_t.permute(1,0,2)
#print(data_t.shape)
out = net(data_t)
print(out.shape)
#while(True): continue;
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_data, 0):
        # get the inputs; data is a list of [inputs, labels]
        #inputs, labels = data
        inputs = data['data'].to(device)#.permute(1,0,2)
        labels = data['label'].to(device)
        #print("Labels: ",labels)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        #print(predicted)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.8f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


    test_func(net,test_data)
#correct_pred = {classname: 0 for classname in classes}
#total_pred = {classname: 0 for classname in classes}
#
## again no gradients needed
#with torch.no_grad():
#    for data in test_data:
#        #images, labels = data
#        inputs = data['data'].to(device)
#        labels = data['label'].to(device)
#        outputs = net(inputs)
#        _, predictions = torch.max(outputs, 1)
#        # collect the correct predictions for each class
#        for label, prediction in zip(labels, predictions):
#            if label == prediction:
#                correct_pred[classes[label]] += 1
#            total_pred[classes[label]] += 1
#
#
## print accuracy for each class
#for classname, correct_count in correct_pred.items():
#    accuracy = 100 * float(correct_count) / total_pred[classname]
#    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
#                                                   accuracy))
print('Finished Training')
