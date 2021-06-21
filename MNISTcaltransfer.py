import random
import torch
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import idx2numpy
import Augmentor
import cv2

'''---------Functions---------'''


def load_image(image, max_size=280, shape=None):
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = in_transform(image).unsqueeze(0)

    return image


def content_augment(aug_image):
    p = Augmentor.Pipeline()
    # p.flip_left_right(probability=0.05)
    # p.flip_top_bottom(probability=0.05)
    # p.crop_random(probability=0.08, percentage_area=0.9)
    # p.shear(probability=0.15 , max_shear_left=5, max_shear_right=20)
    # p.skew_tilt(probability=0.1, magnitude=0.5)
    p.random_distortion(probability=0.7, grid_width=3, grid_height=3, magnitude=5)
    p.skew(probability=0.1, magnitude=0.5)
    p.rotate(probability=0.5, max_left_rotation=20, max_right_rotation=5)
    p.zoom(probability=0.5, min_factor=0.75, max_factor=0.9)
    p.resize(probability=1.0, width=56, height=56)

    in_transform = transforms.Compose([
        p.torch_transform()])
    aug_image = in_transform(aug_image)
    return aug_image


def get_style_image(path):
    imgs_ver = []
    for i in range(0, 10):
        imgs = []
        for j in range(0, 10):
            y = random.randint(0, 9)
            im = Image.open(path + str(y) + '.png').convert('RGB')
            im = content_augment(im)
            imgs.append(im)
        imgs_comb = np.hstack((np.asarray(im) for im in imgs))
        imgs_comb = Image.fromarray(imgs_comb)
        imgs_ver.append(imgs_comb)
    imgs_full = np.vstack((np.asarray(im) for im in imgs_ver))
    imgs_full = Image.fromarray(imgs_full)
    imgs_full.convert('RGB')
    imgs_full = imgs_full.resize((280, 280), Image.ANTIALIAS)
    return imgs_full


def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = image.squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)
    return image


def get_features(image, model):
    layers = {'0': 'conv1_1',
              '5': 'conv2_1',
              '10': 'conv3_1',
              '19': 'conv4_1',
              '21': 'conv4_2',
              '28': 'conv5_1'}
    features = {}

    for name, layer in model._modules.items():
        image = layer(image)
        if name in layers:
            features[layers[name]] = image
    return features


def get_features2(images, model):
    layers = {'0': 'conv1_1',
              '5': 'conv2_1',
              '10': 'conv3_1',
              '19': 'conv4_1',
              '21': 'conv4_2',
              '28': 'conv5_1'}
    features = {}
    for image in images:
        for name, layer in model._modules.items():
            image = layer(image)
            if name in layers:
                features[layers[name]] = image
    return features


def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


'''---------Parameters---------'''

style_weights = {'conv1_1': 0.2,
                 'conv2_1': 0.2,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}
content_weight = 1
style_weight = 50
show_every = 300
steps = 100000
learning_rate = 0.007
nr_style_images = 100
style_path = 'numbers3/'


'''---------Run---------'''

vgg = models.vgg19(pretrained=True).features

for param in vgg.parameters():
    param.requires_grad_(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

file = 'mnist/train-images-idx3-ubyte'
imagearray = idx2numpy.convert_from_file(file)

cont_ver = []
for i in range(0, 10):
    cont = []
    for j in range(0, 10):
        y = random.randint(0, len(imagearray))
        cont.append(imagearray[y])
    cont_comb = np.hstack((np.asarray(im) for im in cont))
    cont_comb = Image.fromarray(cont_comb)
    cont_ver.append(cont_comb)
cont_full = np.vstack((np.asarray(im) for im in cont_ver))
cont_full = Image.fromarray(cont_full)
cont_full = cont_full.convert('RGB')

imgs_full = get_style_image(style_path)

content = load_image(cont_full).to(device)
style = load_image(imgs_full, shape=content.shape[-2:]).to(device)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(im_convert(content))
ax1.axis('off')
ax2.imshow(im_convert(style))
ax2.axis('off')
plt.show()

styles = []
for i in range(nr_style_images):
    stl = load_image(get_style_image(style_path), shape=content.shape[-2:]).to(device)
    styles.append(stl)
content_features = get_features(content, vgg)
style_features = get_features2(styles, vgg)

style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

target = content.clone().requires_grad_(True).to(device)

optimizer = optim.Adam([target], lr=learning_rate)

height, width, channels = im_convert(target).shape
image_array = []
capture_frame = steps / 100
counter = 0

for ii in range(1, steps + 1):
    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        _, d, h, w = target_feature.shape
        style_loss += layer_style_loss / (d * h * w)

    total_loss = content_loss * content_weight + style_loss * style_weight

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if ii % show_every == 0:
        print('Total loss: ', total_loss.item())
        print('Iterations: ', ii)

    if ii % capture_frame == 0:
        image_array.append(im_convert(target))
        counter += 1

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
ax1.imshow(im_convert(content))
ax1.axis('off')
ax2.imshow(im_convert(style))
ax2.axis('off')
ax3.imshow(im_convert(target))
ax3.axis('off')

frame_height, frame_width, _ = im_convert(target).shape
vid = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

for i in range(0, len(image_array)):
    img = image_array[i]
    img = img * 255
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    vid.write(img)

vid.release()

img = im_convert(target)
img = img * 255
img = np.array(img, dtype=np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite('out.jpg', img)

content = im_convert(content)
content = content * 255
content = np.array(content, dtype=np.uint8)
content = cv2.cvtColor(content, cv2.COLOR_BGR2RGB)
cv2.imwrite('mnist.jpg', content)

style = im_convert(style)
style = style * 255
style = np.array(style, dtype=np.uint8)
style = cv2.cvtColor(style, cv2.COLOR_BGR2RGB)
cv2.imwrite('style.jpg', style)
