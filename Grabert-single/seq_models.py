import torch
import torch.nn as nn
import math

class Smiles_embedding(nn.Module):
	def __init__(self, vocab_size, embed_size, max_len, adj=False):
		super().__init__()
		self.token = nn.Embedding(vocab_size, embed_size, padding_idx=0)
		self.position = nn.Embedding(max_len, embed_size)
		self.max_len = max_len
		self.adj = adj
		self.embed_size = embed_size
		if adj:
			self.adj = Adjacency_embedding(max_len, embed_size)

		self.embed_size = embed_size

	def forward(self, sequence, pos_num, adj_mask=None, adj_mat=None):
		x = self.token(sequence) + self.position(pos_num)
		if adj_mat is not None:
			# additional embedding matrix. need to modify
			#print(adj_mask.shape)
			x += adj_mask.unsqueeze(2) * self.adj(adj_mat).repeat(1, self.max_len).reshape(-1,self.max_len, self.embed_size)
		return x

class Adjacency_embedding(nn.Module):
	def __init__(self, input_dim, model_dim, bias=True):
		super(Adjacency_embedding, self).__init__()

		self.weight_h = nn.Parameter(torch.Tensor(input_dim, model_dim))
		self.weight_a = nn.Parameter(torch.Tensor(input_dim))
		if bias:
			self.bias = nn.Parameter(torch.Tensor(model_dim))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight_h.size(1))
		stdv2 = 1. /math.sqrt(self.weight_a.size(0))
		self.weight_h.data.uniform_(-stdv, stdv)
		self.weight_a.data.uniform_(-stdv2, stdv2)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input_mat):
		a_w = torch.matmul(input_mat, self.weight_h)
		out = torch.matmul(a_w.transpose(1,2), self.weight_a)

		if self.bias is not None:
			out += self.bias
		#print(out.shape)
		return out


class BERT_base(nn.Module):
	def __init__(self, model):
		super().__init__()
		self.bert = model
		self.linear = nn.Linear(1024, 64)
	def forward(self, x, pos_num, adj_mask=None, adj_mat=None):
		x = self.bert(x, pos_num, adj_mask, adj_mat)
		x = self.linear(x)
		#print("after linear",x.shape)
		return x

class Smiles_BERT(nn.Module):
	def __init__(self, vocab_size, max_len=256, feature_dim=1024, nhead=4, feedforward_dim=1024, nlayers=6, dropout_rate=0, adj=False):
		super(Smiles_BERT, self).__init__()
		self.embedding = Smiles_embedding(vocab_size, feature_dim, max_len, adj=adj)
		trans_layer = nn.TransformerEncoderLayer(feature_dim, nhead, feedforward_dim, activation='gelu', dropout=dropout_rate)
		self.transformer_encoder = nn.TransformerEncoder(trans_layer, nlayers)
		#print(self.transformer_encoder)
		
		#self.linear = Masked_prediction(feedforward_dim, vocab_size)

	def forward(self, src, pos_num, adj_mask=None, adj_mat=None):
		# True -> masking on zero-padding. False -> do nothing
		#mask = (src == 0).unsqueeze(1).repeat(1, src.size(1), 1).unsqueeze(1)
		mask = (src == 0)
		mask = mask.type(torch.bool)
		#print(mask.shape)

		#print( pos_num.shape, adj_mask.shape, adj_mat.shape)

		x = self.embedding(src, pos_num, adj_mask, adj_mat,)
		x = self.transformer_encoder(x.transpose(1,0), src_key_padding_mask=mask)

		#print("x after transformer_encoder",x.shape)

		x = x.transpose(1,0)
		#x = self.linear(x)
		return x