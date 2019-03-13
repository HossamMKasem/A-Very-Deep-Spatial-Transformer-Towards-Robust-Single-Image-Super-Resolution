-- this program is used to generate a RTS train dataset using Affine transform and bilinear sampler

require 'image';
require 'stn';
--===============================================================================================
-- new program to generate the rotate data set
--===============================================================================================
train_data = torch.load('original_train_dataset.bin')
testimg=train_data.data
label=train_data.label
S4=testimg:size()[3]
cahnnel=testimg:size()[2]
im2=torch.zeros(cahnnel,S4/1,S4/1)
im=torch.zeros(cahnnel,S4/1,S4/1)
output_2=torch.zeros(testimg:size()[1],cahnnel,S4/1,S4/1)
label_temp=torch.zeros(cahnnel,S4,S4)
label_temp_scaled=torch.zeros(cahnnel,S4/1,S4/1)
Final_label=torch.zeros(label:size()[1],cahnnel,S4/1,S4/1)
for iii = 1,testimg:size(1) do
im2 = testimg[iii]
im = image.scale(im2, im2:size()[3] / 1, im2:size()[2] / 1, 'bicubic')
label_temp=label[iii]
label_temp_scaled=image.scale(label_temp, im2:size()[3] / 1, im2:size()[2] / 1, 'bicubic')
Final_label[iii]=label_temp_scaled
width = im:size()[3]  -- 512 / 4
height = im:size()[2]  -- 512 / 4
nchan = im:size()[1]  -- 3 
grid_y = torch.ger( torch.linspace(-1,1,height), torch.ones(width) )
grid_x = torch.ger( torch.ones(height), torch.linspace(-1,1,width) )
flow = torch.FloatTensor()
flow:resize(2,height,width)
flow:zero()
flow_scale = torch.FloatTensor()
flow_scale:resize(2,height,width)
flow_scale[1] = grid_y
flow_scale[2] = grid_x
flow_scale[1]:add(1):mul(0.5) -- 0 to 1
flow_scale[2]:add(1):mul(0.5) -- 0 to 1
flow_scale[1]:mul(height-1)
flow_scale[2]:mul(width-1)
flow:add(flow_scale)
flow_rot = torch.FloatTensor()
flow_rot:resize(2,height,width)
flow_rot[1] = grid_y * ((height-1)/2) * -1
flow_rot[2] = grid_x * ((width-1)/2) * -1
view = flow_rot:reshape(2,height*width)
rot_angle = -20  
r = rot_angle/180*math.pi
rotmat = torch.FloatTensor{{math.cos(r), -math.sin(r)}, {math.sin(r), math.cos(r)}}
flow_rotr = torch.mm(rotmat, view)
flow_rot = flow_rot - flow_rotr:reshape( 2, height, width )
flow:add(flow_rot)
im_bicubic = image.warp(im, flow, 'bicubic', false)

im_bicubic =image.translate(im_bicubic,5,5)

output_2[iii]=im_bicubic

end
dump2={}
dump2.data = output_2:float()
dump2.labels=testimg:float()
torch.save('combined_traindataset_RT.bin',dump2)
v1=dump2.data[38]
v2=dump2.labels[38]
image.save('original_RT.png',image.toDisplayTensor(v2))
image.save('Transformed_RT.png',image.toDisplayTensor(v1))



--####################################################################
-- Scaled Section
--####################################################################

testimg_full = torch.load('combined_traindataset_RT.bin')
testimg=testimg_full.data
testimg_label=testimg_full.labels
S4=testimg:size()[3]
im2=torch.zeros(3,S4,S4)
im=torch.zeros(3,S4,S4)
output_2=torch.zeros(testimg:size()[1],3,S4,S4)


for iii = 1,testimg:size(1) do
im2 = testimg[iii]

im=im2
width = im:size()[3]  
height = im:size()[2] 
nchan = im:size()[1]  
grid_y = torch.ger( torch.linspace(-1,1,height), torch.ones(width) )
grid_x = torch.ger( torch.ones(height), torch.linspace(-1,1,width) )
flow = torch.FloatTensor()
flow:resize(2,height,width)
flow:zero()
flow_scale = torch.FloatTensor()
flow_scale:resize(2,height,width)
flow_scale[1] = grid_y
flow_scale[2] = grid_x
flow_scale[1]:add(1):mul(0.5) -- 0 to 1
flow_scale[2]:add(1):mul(0.5) -- 0 to 1
flow_scale[1]:mul(height-1)
flow_scale[2]:mul(width-1)
flow:add(flow_scale)
flow_rot = torch.FloatTensor()
flow_rot:resize(2,height,width)
flow_rot[1] = grid_y * ((height-1)/2) * -1
flow_rot[2] = grid_x * ((width-1)/2) * -1
view = flow_rot:reshape(2,height*width)
rot_angle = 0  
r = rot_angle/180*math.pi
scale = torch.rand(1)*0.5+0.7
s=scale[1]
rotmat = torch.FloatTensor{{math.cos(r), -math.sin(r)}, {math.sin(r), math.cos(r)}}
rotmat =rotmat:mul(s)
flow_rotr = torch.mm(rotmat, view)
flow_rot = flow_rot - flow_rotr:reshape( 2, height, width )
flow:add(flow_rot)
im_bicubic = image.warp(im, flow, 'bicubic', false)
output_2[iii]=im_bicubic

end

dump2={}
dump2.data= output_2:float()
dump2.label=Final_label:float()
torch.save('combined_traindataset_rotated_original_color.bin',dump2)
v1=dump2.data[38]
v2=dump2.label[38]
image.save('original_RTS.png',image.toDisplayTensor(v2))
image.save('Transformed_RTS.png',image.toDisplayTensor(v1))


