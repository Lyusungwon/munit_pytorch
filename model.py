import torch
from torch import nn
from torch.nn import Sequential
import torch.nn.functional as F

class Content_encoder(nn.Module):
	def __init__(self, channel_size = 64, content_code_h = 44, content_code_w = 54):
		super(Content_encoder, self).__init__()
		self.channel_size = channel_size
		self.content_code_h = content_code_h
		self.content_code_w = content_code_w
		self.conv = Sequential(
			nn.Conv2d(3, self.channel_size, 7, 1, 3),
			nn.ReLU(inplace=True),
			nn.Conv2d(self.channel_size, self.channel_size * 2, 4, 2, 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(self.channel_size * 2, self.channel_size * 4, 4, 2, 1),
			nn.ReLU(inplace=True)
			)
		self.res1 = Sequential(
			nn.Conv2d(self.channel_size * 4, self.channel_size * 4, 3, 1, 1),
			nn.InstanceNorm2d(self.channel_size * 4),
			nn.ReLU(inplace=True),
			nn.Conv2d(self.channel_size * 4, self.channel_size * 4, 3, 1, 1)
			)
		self.res2 = Sequential(
			nn.Conv2d(self.channel_size * 4, self.channel_size * 4, 3, 1, 1),
			nn.InstanceNorm2d(self.channel_size * 4),
			nn.ReLU(inplace=True),
			nn.Conv2d(self.channel_size * 4, self.channel_size * 4, 3, 1, 1)
			)
		self.res3 = Sequential(
			nn.Conv2d(self.channel_size * 4, self.channel_size * 4, 3, 1, 1),
			nn.InstanceNorm2d(self.channel_size * 4),
			nn.ReLU(inplace=True),
			nn.Conv2d(self.channel_size * 4, self.channel_size * 4, 3, 1, 1)
			)
		self.res4 = Sequential(
			nn.Conv2d(self.channel_size * 4, self.channel_size * 4, 3, 1, 1),
			nn.InstanceNorm2d(self.channel_size * 4),
			nn.ReLU(inplace=True),
			nn.Conv2d(self.channel_size * 4, self.channel_size * 4, 3, 1, 1)
			)
		# self.conv2 = nn.Conv2d(self.channel_size * 4, 1, 1)
	def res_block(self, x, res):
		residual = res(x)
		return x + residual

	def forward(self, x):
		x = self.conv(x)
		x = self.res_block(x, self.res1)
		x = self.res_block(x, self.res2)
		x = self.res_block(x, self.res3)
		x = self.res_block(x, self.res4)
		x = x.view(-1, self.channel_size * 4, self.content_code_h, self.content_code_w)
		return x

class Style_encoder(nn.Module):
	def __init__(self, channel_size = 64, style_code_num = 8):
		super(Style_encoder, self).__init__()
		self.channel_size = channel_size
		self.style_code_num = style_code_num
		self.conv = Sequential(
			nn.Conv2d(3, self.channel_size, 7, 1, 3),
			nn.ReLU(inplace=True),
			nn.Conv2d(self.channel_size, self.channel_size * 2, 4, 2, 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(self.channel_size * 2, self.channel_size * 4, 4, 2, 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(self.channel_size * 4, self.channel_size * 4, 4, 2, 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(self.channel_size * 4, self.channel_size * 4, 4, 2, 1),
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(self.channel_size * 4, self.style_code_num, 1)
			)
	def forward(self, x):
		x = self.conv(x)
		x = x.view(-1, self.style_code_num)
		return x

class Res_block(nn.Module):
	def __init__(self, channel_size):
		super(Res_block, self).__init__()	
		self.channel_size = channel_size
		self.conv1 = nn.Conv2d(self.channel_size * 4, self.channel_size * 4, 3, 1, 1)
		self.adain = AdaIN()
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(self.channel_size * 4, self.channel_size * 4, 3, 1, 1)
	def forward(self, x, gamma, beta):
		residual = x
		x = self.relu(self.adain(self.conv1(x), gamma, beta))
		x = self.conv2(x)
		return x + residual

class AdaIN(nn.Module):
	def __init__(self):
		super(AdaIN, self).__init__()	
	def forward(self, x, gamma, beta, use_input_stats=False):
		gamma = gamma.unsqueeze(2).unsqueeze(3)
		beta = beta.unsqueeze(2).unsqueeze(3)
		batch_size = x.size()[0]
		channel_size = x.size()[1]
		x_flatten = x.view(batch_size, channel_size, -1)
		x_mean = x_flatten.mean(2).unsqueeze(2).unsqueeze(3)
		x_std = x_flatten.std(2).unsqueeze(2).unsqueeze(3)
		return gamma * (x - x_mean) / x_std + beta	

class Decoder(nn.Module):
	def __init__(self, output_h = 218, output_w = 178, channel_size = 64, style_code_num = 8):
		super(Decoder, self).__init__()
		self.output_h = output_h
		self.output_w = output_w
		self.channel_size = channel_size
		# self.content_code_num = content_code_num
		self.style_code_num = style_code_num
		self.mlp = Sequential(
			nn.Linear(self.style_code_num, self.channel_size * 4),
			nn.ReLU(inplace=True),
			nn.Linear(self.channel_size * 4, self.channel_size * 4 * 4 ),
			nn.ReLU(inplace=True),
			nn.Linear(self.channel_size * 4 * 4 , self.channel_size * 4 * 4 * 2)
			)
		self.res1 = Res_block(self.channel_size)
		self.res2 = Res_block(self.channel_size)
		self.res3 = Res_block(self.channel_size)
		self.res4 = Res_block(self.channel_size)		

		self.upconv = Sequential(
			nn.Upsample(scale_factor=2),			
			nn.Conv2d(self.channel_size * 4, self.channel_size * 2, 5, 1, 2),
			nn.ReLU(inplace=True),
			nn.Upsample(scale_factor=2),		
			nn.Conv2d(self.channel_size * 2, self.channel_size, 5, 1, 2),
			nn.ReLU(inplace=True),
			nn.Conv2d(self.channel_size, 3, 7, 1, 4)
			)

	def forward(self, c, s):
		s = self.mlp(s)
		s = s.view(-1, 8, self.channel_size * 4)
		c = self.res1(c, s[:,0], s[:,1])
		c = self.res2(c, s[:,2], s[:,3])
		c = self.res3(c, s[:,4], s[:,5])
		c = self.res4(c, s[:,6], s[:,7])
		x = self.upconv(c)
		x = x.view(-1, 3, self.output_h, self.output_w)
		return x

class Discriminator(nn.Module):
	def __init__(self, channel_size = 1):
		super(Discriminator, self).__init__()
		self.channel_size = channel_size
		self.conv = Sequential(
			nn.Conv2d(3, self.channel_size, 4, 2, 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(self.channel_size, self.channel_size * 2, 4, 2, 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(self.channel_size * 2, self.channel_size * 4, 4, 2, 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(self.channel_size * 4, self.channel_size * 8, 4, 2, 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(self.channel_size * 8, 1, 1, 1)
			)
	def forward(self,x):
		x = self.conv(x)
		x = x.mean(3).mean(2)
		x = x.view(-1, 1)
		# x = F.sigmoid(x)
		return x


# def debug():
# 	ce = Content_encoder()
# 	se = Style_encoder()
# 	d = Decoder()
# 	dd = Discriminator()
# 	input = torch.rand(2, 3, 218, 178)
# 	a = ce(input)
# 	print(a.size())
# 	b = se(input)
# 	print(b.size())
# 	# input = torch.rand(5, 8)
# 	c = d(a, b)
# 	print(c.size())
# 	e = dd(c)
# 	print(e.size())

# debug()