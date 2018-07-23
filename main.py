import sys
sys.path.append('../utils/')
import argparser
import dataloader
import model
import time
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

parser = argparser.default_parser()
parser.add_argument('--input-h', type=list, default=218, metavar='N')
parser.add_argument('--input-w', type=list, default=178, metavar='N')
parser.add_argument('--channel-size', type=list, default=64, metavar='N')
parser.add_argument('--content-code-h', type=list, default=44, metavar='N')
parser.add_argument('--content-code-w', type=list, default=54, metavar='N')
parser.add_argument('--style-code-num', type=list, default=8, metavar='N')
parser.add_argument('--lx', type=list, default=1, metavar='N')
parser.add_argument('--lc', type=list, default=1, metavar='N')
parser.add_argument('--ls', type=list, default=1, metavar='N')
args = parser.parse_args()

torch.manual_seed(args.seed)

if args.device == 'cpu':
	device = torch.device('cpu')
else:
	device = torch.device('cuda:{}'.format(args.device))
	torch.cuda.set_device(args.device)

config_list = [args.epochs, args.batch_size, args.lr, 
				args.input_h, args.input_w, 
				args.channel_size, args.content_code_h, args.content_code_w, args.style_code_num,
				args.lx, args.lc, args.ls, args.device]
config = ""
for i in map(str, config_list):
	config = config + '_' + i
print("Config:", config)

train_loader = dataloader.train_loader('celeba', args.data_directory, args.batch_size)
test_loader = dataloader.test_loader('celeba', args.data_directory, args.batch_size)

if args.load_model != '000000000000':
	ce1 = torch.load(args.log_directory + args.load_model + '/content_encoder1.pt')
	ce2 = torch.load(args.log_directory + args.load_model + '/content_encoder2.pt')
	se1 = torch.load(args.log_directory + args.load_model + '/style_encoder1.pt')
	se2 = torch.load(args.log_directory + args.load_model + '/style_encoder2.pt')
	de1 = torch.load(args.log_directory + args.load_model + '/decoder1.pt')
	de2 = torch.load(args.log_directory + args.load_model + '/decoder2.pt')
	dis1 = torch.load(args.log_directory + args.load_model + '/discriminator1.pt')
	dis2 = torch.load(args.log_directory + args.load_model + '/discriminator2.pt')
	args.time_stamep = args.load_mode[:12]
else:
	ce1 = model.Content_encoder(args.channel_size, args.content_code_h, args.content_code_w).to(device)
	ce2 = model.Content_encoder(args.channel_size, args.content_code_h, args.content_code_w).to(device)
	se1 = model.Style_encoder(args.channel_size, args.style_code_num).to(device)
	se2 = model.Style_encoder(args.channel_size, args.style_code_num).to(device)
	de1 = model.Decoder(args.input_h, args.input_w, args.channel_size, args.style_code_num).to(device)
	de2 = model.Decoder(args.input_h, args.input_w, args.channel_size, args.style_code_num).to(device)
	dis1 = model.Discriminator(args.channel_size).to(device)
	dis2 = model.Discriminator(args.channel_size).to(device)

log = args.log_directory + 'munit/' + args.time_stamp + config + '/'
writer = SummaryWriter(log)

gen_optimizer = optim.Adam(list(ce1.parameters()) + list(ce2.parameters()) + 
							list(se1.parameters()) + list(se2.parameters()) +
							list(de1.parameters()) + list(de2.parameters()), lr = args.lr)
dis_optimizer = optim.Adam(list(dis1.parameters()) + list(dis2.parameters()), lr = args.lr)

def train(epoch):
	epoch_start_time = time.time()
	train_loss = 0
	gen_losses = 0
	dis_losses = 0
	ce1.train()
	ce2.train()
	se1.train()
	se2.train()
	de1.train()
	de2.train()
	dis1.train()
	dis2.train()
	fake_style_code = D.Normal(torch.zeros(args.style_code_num).to(device), torch.ones(args.style_code_num).to(device))
	for batch_idx, (input_data1, input_data2) in enumerate(train_loader):
		start_time = time.time()
		batch_size = input_data1.size()[0]
		gen_optimizer.zero_grad()
		input_data1 = input_data1.to(device)
		input_data2 = input_data2.to(device)
		## Within-domain reconstruction
		# 1
		c1 = ce1(input_data1)
		s1 = se1(input_data1)
		recon_data1 = de1(c1, s1)
		irecon_loss1 = F. l1_loss(recon_data1, input_data1, size_average=True)
		# 2
		c2 = ce2(input_data2)
		s2 = se2(input_data2)
		recon_data2 = de2(c2, s2)
		irecon_loss2 = F. l1_loss(recon_data2, input_data2, size_average=True)
		## Cross-domain translation
		# 1
		fake_s1 = fake_style_code.sample(torch.Size([batch_size]))
		output_data21 = de1(c2.detach(), fake_s1)
		recon_c2 = ce1(output_data21)
		recon_s1 = se1(output_data21)
		crecon_loss2 = F. l1_loss(recon_c2, c2.detach(), size_average=True)
		srecon_loss1 = F. l1_loss(recon_s1, fake_s1, size_average=True)
		fake_likelihood1 = dis1(output_data21)
		gen_loss1 = F.mse_loss(fake_likelihood1, torch.ones(batch_size, 1).to(device))

		# 2
		fake_s2 = fake_style_code.sample(torch.Size([batch_size]))
		output_data12 = de1(c1.detach(), fake_s2)
		recon_c1 = ce2(output_data12)
		recon_s2 = se2(output_data12)
		crecon_loss1 = F. l1_loss(recon_c1, c1.detach(), size_average=True)
		srecon_loss2 = F. l1_loss(recon_s2, fake_s2, size_average=True)
		fake_likelihood2 = dis2(output_data12)
		gen_loss2 = F.mse_loss(fake_likelihood2, torch.ones(batch_size, 1).to(device))

		## total
		gen_loss = args.lx * (irecon_loss1 + irecon_loss2) \
				+ args.lc * (crecon_loss1 + crecon_loss2) \
				+ args.ls * (srecon_loss1 + srecon_loss2) \
				+ gen_loss1 + gen_loss2
		gen_losses += gen_loss.item()
		gen_loss.backward(retain_graph=True)
		gen_optimizer.step()

		## discriminator
		# 1
		dis_optimizer.zero_grad()
		real_likelihood1 = dis1(input_data1)
		fake_likelihood1 = dis1(output_data21)
		dis_loss1 = F.mse_loss(real_likelihood1, torch.ones(batch_size, 1).to(device))	\
					+  F.mse_loss(fake_likelihood1, torch.zeros(batch_size, 1).to(device))		
		# 2
		real_likelihood2 = dis2(input_data2)
		fake_likelihood2 = dis2(output_data12)
		dis_loss2 = F.mse_loss(real_likelihood2, torch.ones(batch_size, 1).to(device))	\
					+  F.mse_loss(fake_likelihood2, torch.zeros(batch_size, 1).to(device))		

		# total
		dis_loss = dis_loss1 + dis_loss2
		dis_losses += dis_loss.item()
		dis_loss.backward()
		dis_optimizer.step()

		train_loss += gen_loss.item() + dis_loss.item()

		writer.add_scalars('Detail loss', {'irecon loss1': irecon_loss1,
											'irecon loss2': irecon_loss2,
											'crecon loss1': crecon_loss1,
											'crecon loss2': crecon_loss2,
											'srecon loss1': srecon_loss1,
											'srecon loss2': srecon_loss2,
											'generator loss1': gen_loss1,
											'generator loss2': gen_loss2,
											'discriminator loss1': dis_loss1,
											'discriminator loss2': dis_loss2}, epoch * batch_size + batch_idx)
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.6f}'.format(
				epoch, batch_idx * batch_size, len(train_loader.dataset),
				100. * batch_idx / len(train_loader), gen_loss.item() + dis_loss.item(), time.time() - start_time))

			n = min(batch_size, 8)
			comparison = torch.cat([input_data1[:n],
								  input_data2[:n],
								  recon_data1[:n],
								  recon_data2[:n],
								  output_data12[:n],
								  output_data21[:n]])
			writer.add_image('Train image', comparison, epoch * batch_size + batch_idx)

	print('====> Epoch: {} Average loss: {:.4f}\tTime: {:.4f}'.format(
		epoch, train_loss / len(train_loader.dataset), time.time() - epoch_start_time))	

	writer.add_scalars('Train loss', {'Generator loss': gen_losses / len(train_loader),
											'Discriminator loss': dis_losses / len(train_loader),
											'Train loss': train_loss / len(train_loader)}, epoch)

def test(epoch):
	ce1.eval()
	ce2.eval()
	se1.eval()
	se2.eval()
	de1.eval()
	de2.eval()
	dis1.eval()
	dis2.eval()
	test_loss = 0
	gen_losses = 0
	dis_losses = 0
	fake_style_code = D.Normal(torch.zeros(args.style_code_num).to(device), torch.ones(args.style_code_num).to(device))
	for batch_idx, (input_data1, input_data2) in enumerate(test_loader):
		batch_size = input_data1.size()[0]
		input_data1 = input_data1.to(device)
		input_data2 = input_data2.to(device)
		c1 = ce1(input_data1)
		s1 = se1(input_data1)
		recon_data1 = de1(c1, s1)
		irecon_loss1 = F. l1_loss(recon_data1, input_data1, size_average=True)
		# 2
		c2 = ce2(input_data2)
		s2 = se2(input_data2)
		recon_data2 = de2(c2, s2)
		irecon_loss2 = F. l1_loss(recon_data2, input_data2, size_average=True)
		## Cross-domain translation
		# 1
		fake_s1 = fake_style_code.sample(torch.Size([batch_size]))
		output_data21 = de1(c2.detach(), fake_s1)
		recon_c2 = ce1(output_data21)
		recon_s1 = se1(output_data21)
		crecon_loss2 = F. l1_loss(recon_c2, c2.detach(), size_average=True)
		srecon_loss1 = F. l1_loss(recon_s1, fake_s1, size_average=True)
		fake_likelihood1 = dis1(output_data21)
		gen_loss1 = F.mse_loss(fake_likelihood1, torch.ones(batch_size, 1).to(device))

		# 2
		fake_s2 = fake_style_code.sample(torch.Size([batch_size]))
		output_data12 = de1(c1.detach(), fake_s2)
		recon_c1 = ce2(output_data12)
		recon_s2 = se2(output_data12)
		crecon_loss1 = F. l1_loss(recon_c1, c1.detach(), size_average=True)
		srecon_loss2 = F. l1_loss(recon_s2, fake_s2, size_average=True)
		fake_likelihood2 = dis2(output_data12)
		gen_loss2 = F.mse_loss(fake_likelihood2, torch.ones(batch_size, 1).to(device))

		## total
		gen_loss = args.lx * (irecon_loss1 + irecon_loss2) \
				 + args.lc * (crecon_loss1 + crecon_loss2) \
				 +  args.ls * (srecon_loss1 + srecon_loss2) \
				 + gen_loss1 + gen_loss2 
		gen_losses += gen_loss.item()

		## discriminator
		# 1
		dis_optimizer.zero_grad()
		real_likelihood1 = dis1(input_data1)
		fake_likelihood1 = dis1(output_data21)
		dis_loss1 = F.mse_loss(real_likelihood1, torch.ones(batch_size, 1).to(device))	\
					+  F.mse_loss(fake_likelihood1, torch.zeros(batch_size, 1).to(device))		
		# 2
		real_likelihood2 = dis2(input_data2)
		fake_likelihood2 = dis2(output_data12)
		dis_loss2 = F.mse_loss(real_likelihood2, torch.ones(batch_size, 1).to(device))	\
					+  F.mse_loss(fake_likelihood2, torch.zeros(batch_size, 1).to(device))		

		# total
		dis_loss = dis_loss1 + dis_loss2
		dis_losses += dis_loss.item()

		test_loss += gen_loss + dis_loss

		if batch_idx == args.log_interval:
			n = min(batch_size, 8)
			comparison = torch.cat([input_data1[:n],
								  input_data2[:n],
								  recon_data1[:n],
								  recon_data2[:n],
								  output_data12[:n],
								  output_data21[:n]])
			writer.add_image('Test Image', comparison, epoch * batch_size + batch_idx)
	print('====> Test set loss: {:.4f}'.format(test_loss/ len(test_loader.dataset)))
	writer.add_scalars('Test loss', {'Generator loss': gen_loss / len(test_loader.dataset),
											'Discriminator loss': dis_loss / len(test_loader.dataset),
											'Test loss': test_loss / len(test_loader.dataset)}, epoch)

for epoch in range(args.epochs):
	train(epoch)
	# test(epoch)
	torch.save(ce1, log + 'content_encoder1.pt')
	torch.save(ce2, log + 'content_encoder2.pt')
	torch.save(se1, log + 'style_encoder1.pt')
	torch.save(se2, log + 'style_encoder2.pt')
	torch.save(de1, log + 'decoder1.pt')
	torch.save(de2, log + 'decoder2.pt')
	torch.save(dis1, log + 'discriminator1.pt')
	torch.save(dis2, log + 'discriminator2.pt')
	print('Model saved in ', log)
writer.close()