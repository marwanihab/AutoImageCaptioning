import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        '''
            we need 4 parts to define our decoder we need:
                1. Embedding layer 
                2. The LTSM layer  
                3. Linear Layer 
               
        '''
        super(DecoderRNN, self).__init__()
        
        # Embedding layer 
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        
        # LTSM layer 
        self.lstm_layer = nn.LSTM(input_size = embed_size, hidden_size = hidden_size, num_layers = num_layers, dropout = 0,                                         batch_first=True)
        
        # Linear Layer 
        self.fc1 = nn.Linear(hidden_size, vocab_size)
        
        # Initializing the weights
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.embedding_layer.weight)
        
        
    
    def forward(self, features, captions):
        
        captions = captions[:, :-1] 
        
        embeded_captions = self.embedding_layer(captions)
        
        features = features.unsqueeze(1)
        
        concat_inputs = torch.cat((features, embeded_captions), 1)
        
        lstm_outputs, _ = self.lstm_layer(concat_inputs)
        
        outputs = self.fc1(lstm_outputs)
        
        return outputs
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predicted_sentence = []
        counter = 0
        predicted_idx = None
        
        while counter < max_len:
            
            lstm_outputs, states = self.lstm_layer(inputs, states)
            
            final_outputs = self.fc1(lstm_outputs)
            
            prediction = torch.argmax(final_outputs, dim=2)
            
            predicted_idx = prediction.item()
            
            predicted_sentence.append(predicted_idx)
            
            if (predicted_idx == 1):
                break
            
            inputs = self.embedding_layer(prediction)
            
            counter+=1
         
        return predicted_sentence
            
            
        
        