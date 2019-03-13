require 'torch'
require 'nn'
require 'image'

torch.setdefaulttensortype('torch.FloatTensor')

local networks = {}

-- These are the basic modules used when creating any macro-module
-- Can be modified to use for example cudnn
networks.modules = {}
networks.modules.convolutionModule = nn.SpatialConvolutionMM
networks.modules.poolingModule = nn.SpatialMaxPooling
networks.modules.nonLinearityModule = nn.ReLU

-- Size of the input image. This comes from the dataset loader

networks.base_input_size =48 

-- Number of output classes. This comes from the dataset.
--networks.nbr_classes = 43

-- Creates a conv module with the specified number of channels in input and output
-- If multiscale is true, the total number of output channels will be:
-- nbr_input_channels + nbr_output_channels
-- Using no_cnorm removes the spatial contrastive normalization module
-- The filter size for the convolution can be specified (default 5)
-- The stride of the convolutions is fixed at 1
function networks.new_conv(nbr_input_channels,nbr_output_channels,
                           multiscale, no_cnorm, filter_size)
  multiscale = multiscale or false
  no_cnorm = no_cnorm or false
  filter_size = filter_size or 5
  local padding_size = 2
  local pooling_size = 2
  local normkernel = image.gaussian1D(7)

  local conv

  local first = nn.Sequential()
  first:add(networks.modules.convolutionModule(nbr_input_channels,
                                      nbr_output_channels,
                                      filter_size, filter_size,
                                      1,1,
                                      padding_size, padding_size))
  first:add(networks.modules.nonLinearityModule())
  first:add(networks.modules.poolingModule(pooling_size, pooling_size,
                                           pooling_size, pooling_size))
  if not no_cnorm then
    first:add(nn.SpatialContrastiveNormalization(nbr_output_channels,
                                                 norm_kernel))
  end

  if multiscale then
    conv = nn.Sequential()
    local second = networks.modules.poolingModule(pooling_size, pooling_size,
                                              pooling_size, pooling_size)

    local parallel = nn.ConcatTable()
    parallel:add(first)
    parallel:add(second)
    conv:add(parallel)
    conv:add(nn.JoinTable(1,3))
  else
    conv = first
  end

  return conv
end
function networks.new_conv2(nbr_input_channels,nbr_output_channels,
                           multiscale, no_cnorm, filter_size)
  multiscale = multiscale or false
  no_cnorm = no_cnorm or false
  filter_size = filter_size or 3
  local padding_size = 1
  local pooling_size = 1
  local normkernel = image.gaussian1D(7)

  local conv

  local first = nn.Sequential()
  first:add(networks.modules.convolutionModule(nbr_input_channels,
                                      nbr_output_channels,
                                      filter_size, filter_size,
                                      1,1,
                                      padding_size, padding_size))
  first:add(networks.modules.nonLinearityModule())
  --first:add(networks.modules.poolingModule(pooling_size, pooling_size,
    --                                       pooling_size, pooling_size))
  if not no_cnorm then
    first:add(nn.SpatialContrastiveNormalization(nbr_output_channels,
                                                 norm_kernel))
  end

  if multiscale then
    conv = nn.Sequential()
    local second = networks.modules.poolingModule(pooling_size, pooling_size,
                                              pooling_size, pooling_size)

    local parallel = nn.ConcatTable()
    parallel:add(first)
    parallel:add(second)
    conv:add(parallel)
    conv:add(nn.JoinTable(1,3))
  else
    conv = first
  end

  return conv
end



function networks.convs_noutput(convs, input_size)
  input_size = input_size or networks.base_input_size
  -- Get the number of channels for conv that are multiscale or not
  local nbr_input_channels = convs[1]:get(1).nInputPlane or
                             convs[1]:get(1):get(1).nInputPlane
  local output = torch.Tensor(1,nbr_input_channels, input_size, input_size)
  for _, conv in ipairs(convs) do
    output = conv:forward(output)
  end
  return output:nElement(), output:size(3)
end

-- Creates a fully connection layer with the specified size.
function networks.new_fc(nbr_input, nbr_output)
  local fc = nn.Sequential()
  fc:add(nn.View(nbr_input))
  fc:add(nn.Linear(nbr_input, nbr_output))
  fc:add(networks.modules.nonLinearityModule())
  return fc
end

-- Creates a classifier with the specified size.
function networks.new_classifier(nbr_input, nbr_output)
  local classifier = nn.Sequential()
  classifier:add(nn.View(nbr_input))
  classifier:add(nn.Linear(nbr_input, nbr_output))
  return classifier
end

-- Creates a spatial transformer module
-- locnet are the parameters to create the localization network
-- rot, sca, tra can be used to force specific transformations
-- input_size is the height (=width) of the input
-- input_channels is the number of channels in the input
-- no_cuda due to (1) below, we need to know if the network will run on cuda


function networks.new_spatial_tranformer_new(locnet, rot, sca, tra,
                                         input_size, input_channels,
                                         no_cuda)
  input_size = input_size or networks.base_input_size
  input_channels = input_channels or 3
  require 'stn'
  local nbr_elements = {}
  for c in string.gmatch(locnet, "%d+") do
    nbr_elements[#nbr_elements + 1] = tonumber(c)
  end


  -- Get number of params and initial state
  local init_bias = {}
  local nbr_params = 0
  if rot then
    nbr_params = nbr_params + 1
    init_bias[nbr_params] = 0
  end
  if sca then
    nbr_params = nbr_params + 1
    init_bias[nbr_params] = 1
  end
  if tra then
    nbr_params = nbr_params + 2
    init_bias[nbr_params-1] = 0
    init_bias[nbr_params] = 0
  end
  if nbr_params == 0 then
    -- fully parametrized case
    nbr_params = 6
    init_bias = {1,0,0,0,1,0}
  end
  print ('nbr_params==')
print(nbr_params)
print ('init_bias==')
print (init_bias)
  local st = nn.Sequential()

   -- Create a localization network same as cnn but with downsampled inputs
  local localization_network = nn.Sequential()
   local branch3=nn.Sequential()
   local branch4=nn.Identity()
   local CVPR_network1=nn.ConcatTable()
   local CVPR_network=nn.Sequential()
  local conv1 = networks.new_conv2(3, 64,
                                           	 false, false, 3)
	local conv2 = networks.new_conv2(64, 64,
                                           	 false, false, 3)										 

	local conv3 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv4 = networks.new_conv2(64, 64,
                                           	 false, false, 3)										 
	local conv5 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv6 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv7 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv8 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv9 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv10 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv11 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv12 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv13 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv14 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv15 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv16 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv17 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv18 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv19 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv20 = networks.new_conv2(64, 3,
                                           	 false, false, 3)											 
											 
  local conv_output_size = networks.convs_noutput({conv20}, input_size/2)
 -- local conv_output_size = networks.convs_noutput({conv3}, input_size/2)
  local fc = networks.new_fc(conv_output_size, nbr_elements[3])
  local classifier = networks.new_classifier(nbr_elements[3], nbr_params)
  -- Initialize the localization network (see paper, A.3 section)
  classifier:get(2).weight:zero()
  classifier:get(2).bias = torch.Tensor(init_bias)


  localization_network:add(networks.modules.poolingModule(2,2,2,2))
    branch3:add(conv1)
    branch3:add(conv2)
    branch3:add(conv3)
    branch3:add(conv4)
	branch3:add(conv5)
    branch3:add(conv6)
    branch3:add(conv7)
	branch3:add(conv8)
	branch3:add(conv9)
	branch3:add(conv10)
	branch3:add(conv11)
	branch3:add(conv12)
	branch3:add(conv13)
	branch3:add(conv14)
	branch3:add(conv15)
	branch3:add(conv16)
	branch3:add(conv17)
	branch3:add(conv18)
	branch3:add(conv19)
	branch3:add(conv20)
	CVPR_network1:add(branch3)
	CVPR_network1:add(branch4)
	CVPR_network:add(CVPR_network1)
	CVPR_network:add(nn.CAddTable())
	
    localization_network:add(CVPR_network)
    localization_network:add(fc)
    localization_network:add(classifier)


  -- Create the actual module structure
  local ct = nn.ConcatTable()
  local branch1 = nn.Sequential()
  branch1:add(nn.Transpose({3,4},{2,4}))
  if not no_cuda then -- see (1) below
    branch1:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
  end
  local branch2 = nn.Sequential()
  branch2:add(localization_network)
  branch2:add(nn.AffineTransformMatrixGenerator(rot, sca, tra))

  branch2:add(nn.AffineGridGeneratorBHWD(input_size,input_size))
  if not no_cuda then -- see (1) below
    branch2:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
  end
  ct:add(branch1)
  ct:add(branch2)

  st:add(ct)
  local sampler = nn.BilinearSamplerBHWD()
  -- (1)
  -- The sampler lead to non-reproducible results on GPU
  -- We want to always keep it on CPU
  -- This does no lead to slowdown of the training
  if not no_cuda then
    sampler:type('torch.FloatTensor')
    -- make sure it will not go back to the GPU when we call
    -- ":cuda()" on the network later
    sampler.type = function(type)
      return self
    end
    st:add(sampler)
    st:add(nn.Copy('torch.FloatTensor','torch.CudaTensor', true, true))
  else
    st:add(sampler)
  end
  st:add(nn.Transpose({2,4},{3,4}))

  return st
  
end

-- Main factory function
-- Can take an opt.net file that will be used to create the network
-- Can take an opt.cnn option that specify how to build the network.
function networks.new(opt)
  local network
  network = nn.Sequential()
  local network2=nn.ConcatTable()
  
   local conv1 = networks.new_conv2(3, 64,
                                           	 false, false, 3)
	local conv2 = networks.new_conv2(64, 64,
                                           	 false, false, 3)										 

	local conv3 = networks.new_conv2(64,64,
                                           	 false, false, 3)
	local conv4 = networks.new_conv2(64, 64,
                                           	 false, false, 3)										 
	local conv5 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv6 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv7 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv8 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv9 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv10 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv11 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv12 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv13 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv14 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv15 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv16 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv17 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv18 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv19 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv20 = networks.new_conv2(64, 3,
                                           	 false, false, 3)	
											 
											 
	  
	  local branch3=nn.Sequential()
	 
									
    branch3:add(conv1)
	branch3:add(conv2)
	branch3:add(conv3)
	branch3:add(conv4)
	branch3:add(conv5)
	branch3:add(conv6)
	branch3:add(conv7)
	branch3:add(conv8)
	branch3:add(conv9)
	branch3:add(conv10)
	branch3:add(conv11)
	branch3:add(conv12)
	branch3:add(conv13)
	branch3:add(conv14)
	branch3:add(conv15)
	branch3:add(conv16)
	branch3:add(conv17)
	branch3:add(conv18)
	branch3:add(conv19)
	branch3:add(conv20)
	
	local branch4= nn.Sequential()
	branch4:add(nn.Identity())
	
  network2:add(branch3)
  network2:add(branch4)
  
    network:add(network2)
  network:add(nn.CAddTable())
  network:add(networks.new_spatial_tranformer_new(opt.locnet,
                                                  opt.rot, opt.sca, opt.tra,
                                                  nil, nil,
                                                  opt.no_cuda))

  return network
end

return networks

