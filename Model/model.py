# -*- coding: utf-8 -*-


import torch.nn as nn
from transformers import DistilBertModel
import torch



class TwoLayerBiLSTM(nn.Module):
    def __init__(self, input_size=768, hidden_size=64, num_layers=2, cnn_out_channels=128, kernel_size=3):
        super(TwoLayerBiLSTM, self).__init__()

        # Bidirectional LSTM
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers,
                              batch_first=True, bidirectional=True)

        # Layer normalization on the hidden states
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # 1D CNN layer for aggregation
        self.cnn = nn.Conv1d(in_channels=hidden_size * 2,
                             out_channels=cnn_out_channels,
                             kernel_size=3,stride = 1,
                             padding=kernel_size // 2)


    def forward(self, x):
        # Pass through BiLSTM
        bilstm_out, _ = self.bilstm(x)

        # Apply layer normalization across the bidirectional hidden states
        bilstm_out = self.layer_norm(bilstm_out)

        # Rearrange dimensions for CNN (batch, features, sequence_length)
        bilstm_out = bilstm_out.permute(0, 2, 1)

        # Pass through 1D CNN layer
        cnn_out = self.cnn(bilstm_out)

        # Permute back to (batch, sequence_length, cnn_out_channels)
        final_out = cnn_out

        return final_out


def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding='valid'),nn.BatchNorm1d(out_channels),
                                   nn.ReLU(inplace=True), nn.MaxPool1d(2,1))
class AudioCNNPool(nn.Module):
    def __init__(self, num_classes=3):
        super(AudioCNNPool, self).__init__()

        input_channels = 25
        self.conv1d_0 = conv1d_block_audio(input_channels, 64)
        self.conv1d_1 = conv1d_block_audio(64, 128)
        self.conv1d_2 = conv1d_block_audio(128, 256)
        self.conv1d_3 = conv1d_block_audio(256, 128)
        self.classifier_1 = nn.Sequential(nn.Linear(128, num_classes))
    def forward(self, x):
        return x
    def forward_stage1(self,x):
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x
    def forward_stage2(self,x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        return x



class MultiModal(nn.Module):
    def __init__(self, num_classes=3, fusion='ia', seq_length=18, num_heads=1):
        super(MultiModal, self).__init__()

        self.audio_model = AudioCNNPool(num_classes=num_classes)
        self.text_model =  TwoLayerBiLSTM()

        #self.pe_a = PositionalEncoding(156, dropout=0.1)

        e_dim = 128
        input_dim_text =  355
        input_dim_audio = 128

        self.av1 = Attention(in_dim_k=input_dim_text, in_dim_q=input_dim_audio, out_dim=input_dim_audio, num_heads=num_heads)
        self.va1 = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_text, out_dim=input_dim_text, num_heads=num_heads)

        self.classifier_1 = nn.Sequential(
                    nn.Linear(e_dim*2, num_classes),
                )

    def forward(self, x_audio, x_text):

        #x_audio = self.pe_a(x_audio)


        x_audio = self.audio_model.forward_stage1(x_audio)
        #print(x_audio.size())
        #print(x_text.size())
        #x_text = self.text_model.forward1(x_text)

        proj_x_a = x_audio.permute(0,2,1)
        proj_x_t = x_text.permute(0,2,1)
        #print("after permutation")
        #print(proj_x_a.size())
        #print(x_text.size())
        #print(proj_x_a.size(), proj_x_t.size())
        _, h_at = self.av1(proj_x_t, proj_x_a)
        _, h_ta = self.va1(proj_x_a, proj_x_t)
        if h_at.size(1) > 1:
            h_at = torch.mean(h_at, axis=1).unsqueeze(1)
        h_at = h_at.sum([-2])
        if h_ta.size(1) > 1:
            h_ta = torch.mean(h_ta, axis=1).unsqueeze(1)
        h_ta = h_ta.sum([-2])

        x_audio = h_ta*x_audio
        x_text = h_at*x_text

        #x_text = x_text.permute(1,0,2)




        x_audio = self.audio_model.forward_stage2(x_audio)
        x_text = self.text_model.forward(x_text)


        audio_pooled = x_audio.mean([-1])
        text_pooled = x_text.mean([-1])

        x = torch.cat((audio_pooled, text_pooled), dim=-1)
        x = self.classifier_1(x)
        return x
