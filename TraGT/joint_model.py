# import torch
# import torch.nn as nn
# from seq_models import TransformerModel
# from graph_models import Graph_Transformer
# from fusion_model import FusionModel


# class Model(torch.nn.Module):
#     def __init__(self, args):
#         super(Model, self).__init__()
#         self.option=args[0]
#         self.graph=args[0][0]
#         self.sequence=args[0][1]
#         self.fusion=args[0][2]
#         self.device=args[1]
#         self.train_dataset=args[2]
#         self.test_dataset=args[3]
#         self.learning_rate=args[4]
#         self.input_dim_train=args[5]
#         self.input_dim_test=args[6]
        
#         vocab_size = 100
#         d_model = 100
#         nhead = 4
#         num_encoder_layers = 3
#         dim_feedforward = 512

#         self.graph_model = Graph_Transformer(in_channels=self.input_dim_train, hidden_channels=64, out_channels=1, heads=4)
#         self.sequence_model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward)
#         self.fusion_model = FusionModel(self.graph_model, self.sequence_model)
#         self.optimizer = torch.optim.Adam(self.fusion_model.parameters(), lr=self.learning_rate)
#         self.loss_fn = nn.BCEWithLogitsLoss()
        
#     def train(self, graph_data, sequence_data):
#         if self.graph:
#             graph_embedding = self.graph_model(graph_data)
#             output = graph_embedding
        
#         if self.sequence:
#             sequence_embedding = self.sequence_model(sequence_data)
#             output = sequence_embedding
        
#         if self.fusion:
#             output = self.fusion_model(graph_data, sequence_data)


#         loss = self.loss_fn(output, graph_data.y.view(-1, 1))
        
#         loss.backward()
        
#         self.optimizer.step()
#         return output, loss