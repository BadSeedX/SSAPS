import cv2
import torch
from PIL import Image
from pylab import *
import random

# get ground truth
def one_hot(data):

    height, width = data.shape[0], data.shape[1]
    data = data.astype('int')
    label = np.zeros((2, height, width), dtype=np.int)
    label[0:1, :, :] = data
    label[1:2, :, :] = 1 - label[0, :, :]
    label = label.astype('float32')
    return label

def cut_paste_collate_fn(batch):
    img_types = list(zip(*batch))
    return [torch.stack(imgs) for imgs in img_types]

# expansion param
def get_pads(height, width, x, y):
    upper_bound, lower_bound = 0.5, 0.9
    lam1, lam2 = random.uniform(upper_bound, lower_bound), random.uniform(upper_bound, lower_bound)
    lam3, lam4 = random.uniform(upper_bound, lower_bound), random.uniform(upper_bound, lower_bound)
    lam5, lam6 = random.uniform(upper_bound, lower_bound), random.uniform(upper_bound, lower_bound)
    lam7, lam8 = random.uniform(upper_bound, lower_bound), random.uniform(upper_bound, lower_bound)
    pads = [[(height/2-x)*lam1, (height/2-x)*(1-lam1), (width/2-y)*lam2, (width/2-y)*(1-lam2)],
            [(height/2-x)*lam3, (height/2-x)*(1-lam3), (width/2-y)*(1-lam4), (width/2-y)*lam4],
            [(height/2-x)*(1-lam5), (height/2-x)*lam5, (width/2-y)*lam6, (width/2-y)*(1-lam6)],
            [(height/2-x)*(1-lam7), (height/2-x)*lam7, (width/2-y)*(1-lam8), (width/2-y)*lam8]]    # 上下左右
    return pads


class CutPaste(object):
    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, org_img, img, gt_org, gt_mic):
        if self.transform:
            org_img = self.transform(org_img)
            img = self.transform(img)

        return org_img, img, gt_org, gt_mic

# construct augmented samples
class AugPatch(CutPaste):
    def __init__(self, **kwags):
        super(AugPatch, self).__init__(**kwags)

    def __call__(self, img):

        origin_image = img.copy()
        origin_image = cv2.cvtColor(np.asarray(origin_image), cv2.COLOR_RGB2BGR)

        height, width, channel = origin_image.shape
        copy_raw = np.zeros((height, width, channel), np.uint8)  # mask image
        copy_image = origin_image.copy()

        # number of patch; size of patch; irregularity of patch
        outlines = []
        size = 163
        for i in range(4):
            outline = []
            (x_boarder, y_boarder) = np.random.randint(0, 384-size, size=2)
            its = np.random.randint(4, 12)
            for it in range(its):
                x = np.random.randint(x_boarder, x_boarder + size)
                y = np.random.randint(y_boarder, y_boarder + size)
                outline.append([x, y])
            outlines.append(outline)
        direction = [[0, 0], [0, width // 2], [height // 2, 0], [height // 2, width // 2]]
        index = 0

        mask_images = []
        for outline in outlines:
            x_coords, y_coords = [item[1] for item in outline], [item[0] for item in outline]
            x_min, x_max, y_min, y_max = np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)

            polygon = np.array(outline)
            copy = copy_raw.copy()
            cv2.fillPoly(copy, [polygon], (255, 255, 255))  # The polygon part turns white
            mask_index = list(np.where(copy > 0))
            copy[tuple(mask_index)] = origin_image[tuple(mask_index)]  # Add mask with I

            mask = copy[x_min:x_max, y_min:y_max]

            pads = get_pads(height, width, abs(x_max - x_min), abs(y_max - y_min))
            up_pad, down_pad, left_pad, right_pad = pads[index][0], pads[index][1], pads[index][2], pads[index][3]

            mask_image = cv2.copyMakeBorder(mask, int(up_pad), int(down_pad), int(left_pad), int(right_pad),
                                            cv2.BORDER_CONSTANT, value=[0, 0, 0])
            mask_image = cv2.resize(mask_image, (height // 2, width // 2))
            mask_images.append(mask_image)  # Process mask polygon position

            x_coords_ = [int(x_coord + pads[index][0] - x_min) for x_coord in x_coords]
            y_coords_ = [int(y_coord + pads[index][2] - y_min) for y_coord in y_coords]
            outline_ = []
            for i in range(len(x_coords_)):
                outline_.append([y_coords_[i] + direction[index][1], x_coords_[i] + direction[index][0]])
            polygon_ = np.array(outline_)
            cv2.fillPoly(copy_image, [polygon_], (0, 0, 0))  # crop p in I

            index += 1

        top_half = cv2.hconcat([mask_images[0], mask_images[1]])
        bottom_half = cv2.hconcat([mask_images[2], mask_images[3]])
        result_image = cv2.vconcat([top_half, bottom_half])
        result = cv2.add(result_image, copy_image)
        result = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

        gt_mic = copy_image.copy()
        gt_mic[gt_mic != 0] = 255
        white_index = list(np.where(gt_mic == 0))
        black_index = list(np.where(gt_mic == 255))
        gt_mic[tuple(white_index)] = 255
        gt_mic[tuple(black_index)] = 0

        gt_org = np.zeros_like(copy_image)

        gt_org = cv2.cvtColor(gt_org, cv2.COLOR_BGR2GRAY)
        gt_mic = cv2.cvtColor(gt_mic, cv2.COLOR_BGR2GRAY)
        gt_mic = gt_mic.astype('float32') / 255
        gt_org = gt_org.astype('float32') / 255

        gt_mic = one_hot(gt_mic)
        gt_org = one_hot(gt_org)

        return super().__call__(img, result, gt_org, gt_mic)
