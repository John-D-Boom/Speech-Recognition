import torch.nn as nn

class BiLSTM(nn.Module):

    def __init__(self, num_layers, in_dims, hidden_dims, out_dims):
        super().__init__()

        self.lstm = nn.LSTM(in_dims, hidden_dims, num_layers, bidirectional=True)
        self.proj = nn.Linear(hidden_dims * 2, out_dims)

    def forward(self, feat):
        hidden, _ = self.lstm(feat)
        output = self.proj(hidden)
        return output

class BiLSTM_Dropout(nn.Module):
    def __init__(self, num_layers, in_dims, hidden_dims, out_dims, dropout):
        super().__init__()

        self.lstm = nn.LSTM(in_dims, 
                            hidden_dims, 
                            num_layers, 
                            bidirectional=True, 
                            dropout = dropout) #This dropout only applies between LSTM layers

        feed_forward_dropout = 0 #Default to 0 so that dropout only between LSTM layers
        if num_layers == 1: #If only a single LSTM, dropout needs to actaully occur
            feed_forward_dropout = dropout 

        self.drop = nn.Dropout(p = feed_forward_dropout) 
        self.proj = nn.Linear(hidden_dims *2 , out_dims)
        self.num_layers = num_layers

    def forward(self, feat):
        hidden, _ = self.lstm(feat)
        dropped = self.drop(hidden)
        output = self.proj(dropped)
        return output


class BiLSTM_Dropout_Unidir(nn.Module):
    def __init__(self, num_layers, in_dims, hidden_dims, out_dims, dropout):
        super().__init__()

        self.lstm = nn.LSTM(in_dims, 
                            hidden_dims, 
                            num_layers, 
                            bidirectional=False, 
                            dropout = dropout) #This dropout only applies between LSTM layers

        feed_forward_dropout = 0 #Default to 0 so that dropout only between LSTM layers
        if num_layers == 1: #If only a single LSTM, dropout needs to actaully occur
            feed_forward_dropout = dropout 

        self.drop = nn.Dropout(p = feed_forward_dropout) 
        self.proj = nn.Linear(hidden_dims, out_dims)
        self.num_layers = num_layers

    def forward(self, feat):
        hidden, _ = self.lstm(feat)
        dropped = self.drop(hidden)
        output = self.proj(dropped)
        return output
