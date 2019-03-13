require 'nn'
require 'optim'
require 'sys'

-- Load Facebook optim package
paths.dofile('Optim.lua')


local trainer = {}

local PSNR

-- This function should be called before any other on the trainer package.
-- Takes as input a torch network, a criterion and the options of the training
function trainer.initialize(network, criterion, options)
  local optim_state = {
      learningRate = options.lr,
      momentum = options.mom,
      learningRateDecay = options.lrd,
      weightDecay = options.wd,
    }

  trainer.tensor_type = torch.getdefaulttensortype()
  if not options.no_cuda then
    trainer.tensor_type = 'torch.CudaTensor'
  end
  trainer.batch_size = options.bs
  trainer.network = network
  if criterion then
    trainer.criterion = criterion
    trainer.optimizer = nn.Optim(network, optim_state)
  end
end

-- Main training function.
-- This performs one epoch of training on the network given during
-- initialization using the given dataset.
-- Returns the mean error on the dataset.
function trainer.train(dataset,epoch)
  if not trainer.optimizer then
    error('Trainer not initialized properly. Use trainer.initialize first.')
  end
  -- do one epoch
  print('<trainer> on training set:')
  local epoch_error = 0
  local nbr_samples = dataset.data:size(1)
  local size_samples = dataset.data:size()[dataset.data:dim()]
  local time = sys.clock()

  -- generate random training batches
  local indices = torch.randperm(nbr_samples):long():split(trainer.batch_size)
  indices[#indices] = nil -- remove last partial batch

  -- preallocate input and target tensors
  local inputs = torch.zeros(trainer.batch_size, 3,
                                    size_samples, size_samples,
                                    trainer.tensor_type)
local targets = torch.zeros(trainer.batch_size, 3,
                                   size_samples, size_samples,
                                    trainer.tensor_type)
 
 --local targets = torch.zeros(trainer.batch_size, 1,
    --                                  trainer.tensor_type)

  for t,ind in ipairs(indices) do
    -- get the minibatch
    inputs:copy(dataset.data:index(1,ind))
  --local datasetlabel={}
  -- datasetlabel.train_label_bin='train_dataset.bin'
  -- local train_label=torch.load(datasetlabel.train_label_bin)
       targets:copy(dataset.label:index(1,ind))

    epoch_error = epoch_error + trainer.optimizer:optimize(optim.sgd,
                                  inputs,
                                  targets,
                                  trainer.criterion)

    -- disp progress
    xlua.progress(t*trainer.batch_size, nbr_samples)
  end
  -- finish progress
  xlua.progress(nbr_samples, nbr_samples)

  -- time taken
  time = sys.clock() - time
  time = time / nbr_samples
  print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')
  print("<trainer> mean MSE  error (train set) = " .. epoch_error/nbr_samples)
  mean_Time=torch.zeros(10)
  mean_MSE_error_Train=torch.zeros(10)
  mean_Time[epoch]= time*1000
  mean_MSE_error_Train[epoch]=epoch_error/nbr_samples
  
  return epoch_error
end

-- Main testing function.
-- This performs a full test on the given dataset using the network
-- given during the initialization.
-- Returns the mean error on the dataset and the accuracy.
function trainer.test(dataset,epoch)
  if not trainer.network then
    error('Trainer not initialized properly. Use trainer.initialize first.')
  end
  -- test over given dataset
  print('')
  print('<trainer> on testing Set:')
  local time = sys.clock()
  local nbr_samples = dataset.data:size(1)
  local size_samples = dataset.data:size()[dataset.data:dim()]
  local epoch_error = 0
  local correct = 0
  local all = 0

  -- generate indices and split them into batches
  local indices = torch.range(1,nbr_samples):long()
  indices = indices:split(trainer.batch_size)

  -- preallocate input and target tensors
  local inputs = torch.zeros(trainer.batch_size, 3,
                                    size_samples, size_samples,
                                    trainer.tensor_type)
local targets = torch.zeros(trainer.batch_size, 3,
                                    size_samples, size_samples,
                                    trainer.tensor_type)
 
local  output_1=torch.Tensor(nbr_samples,3,48,48)
local  output_2=torch.Tensor(nbr_samples,3,48,48)
local  output_3=torch.Tensor(nbr_samples,3,48,48)
local  output_5=torch.Tensor(nbr_samples,3,48,48)
local  output_7=torch.Tensor(nbr_samples,3,48,48)
local  output_10=torch.Tensor(nbr_samples,3,48,48)
count_1=1
count_2=1
count_3=1
count_5=1
count_7=1
count_10=1


  for t,ind in ipairs(indices) do
    -- last batch may not be full
    local local_batch_size = ind:size(1)
    -- resize prealocated tensors (should only happen on last batch)
   inputs:resize(local_batch_size,3,size_samples,size_samples)
   targets:resize(local_batch_size, 3,size_samples,size_samples)
    -- targets:resize(local_batch_size, 1)
   inputs:copy(dataset.data:index(1,ind))
   targets:copy(dataset.labels:index(1,ind))
  
    -- test samples
    local scores = trainer.network:forward(inputs)

if  epoch==1 then 
	output_1:narrow(1,count_1,local_batch_size):copy(scores)
     count_1=count_1+local_batch_size
     torch.save('output_1.bin',output_1)
		end	
if  epoch==2 then 
	output_2:narrow(1,count_2,local_batch_size):copy(scores)
     count_2=count_2+local_batch_size
     torch.save('output_2.bin',output_2)
		end	
if  epoch==3 then 
	output_3:narrow(1,count_3,local_batch_size):copy(scores)
     count_3=count_3+local_batch_size
     torch.save('output_3.bin',output_3)
		end	
	
if  epoch==5 then 
	output_5:narrow(1,count_5,local_batch_size):copy(scores)
     count_5=count_5+local_batch_size
     torch.save('output_5.bin',output_5)
		end
	
if  epoch==10 then 
	output_10:narrow(1,count_10,local_batch_size):copy(scores)
     count_10=count_10+local_batch_size
     torch.save('output_10.bin',output_10)
		end
	 

    epoch_error = epoch_error + trainer.criterion:forward(scores,
                                  targets)



    -- disp progress
    xlua.progress(t*trainer.batch_size, nbr_samples)
  end
  -- finish progress
  xlua.progress(nbr_samples, nbr_samples)

  -- timing
  time = sys.clock() - time
  time = time / nbr_samples
  print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')
  print("<trainer> mean MSE error (te st set) = " .. epoch_error/nbr_samples)
  mean_MSE_error_Test=torch.zeros(10)
  mean_Time_Test=torch.zeros(10)
  mean_Time_Test[epoch]=time*1000
  mean_MSE_error_Test[epoch]=epoch_error/nbr_samples
   return epoch_error, accuracy
end



return trainer
