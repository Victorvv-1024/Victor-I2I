import os
from utils import util
import matplotlib.pyplot as plt
import numpy as np



def ag_generate_image(dataloader, netG, opt, device, translated_only=False):
    for idx, raw_data in enumerate(dataloader):

        # mask, data = raw_data[0], raw_data[1]
        data = raw_data
        data = data.to(device)
        # mask = mask.to(device)
        source_img = util.tensor2img(data)
        source_img = (source_img * 127.5 + 127.5).astype(np.uint8)
        # translated
        translated, translated_mask, translated_content = netG(data)
        translated = util.tensor2img(translated)
        translated = (translated * 127.5 + 127.5).astype(np.uint8)

        if translated_only:
            plt.figure(figsize=(20, 20), dpi=100)
            plt.axis('off')
            plt.imshow(translated)
            # plt.imsave(f'{save_path}/infer={idx + 1}.png', translated)
        else:
            titles = ['Source', "Translated", "Attention mask", "Content Mask"]
            _, ax = plt.subplots(1, len(titles), figsize=(20, 20))
            [ax[i].set_title(title) for i, title in enumerate(titles)]
            translated_mask = util.tensor2img(translated_mask, isMask=True)

            translated_content = util.tensor2img(translated_content)
            translated_content = (translated_content * 127.5 + 127.5).astype(np.uint8)

            [ax[j].imshow(img) for j, img in enumerate([source_img, translated, translated_mask, translated_content])]
            [ax[j].axis("off") for j in range(len(titles))]


        plt.savefig(os.path.join(opt.out_dir, f'infer={idx + 1}.png'), bbox_inches='tight', dpi=100)
        plt.close()