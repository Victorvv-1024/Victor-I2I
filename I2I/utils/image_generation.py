import os
from utils import util
import matplotlib.pyplot as plt
import numpy as np



def generate_image(dataloader, netG, opt, device, translated_only=False):
    # check if it is ag
    ag = False
    if 'ag' in opt.model:
        ag = True 
    
    for idx, (raw_data,fname) in enumerate(dataloader):      
        # mask, data = raw_data[0], raw_data[1]
        data = raw_data
        data = data.to(device)
        # mask = mask.to(device)
        source_img = util.tensor2img(data)
        source_img = (source_img * 127.5 + 127.5).astype(np.uint8)
        
        # translation
        if ag:
            translated, translated_mask, translated_content = netG(data)
            translated = util.tensor2img(translated)
            translated = (translated * 127.5 + 127.5).astype(np.uint8)
        else:
            translated = netG(data)
            translated = util.tensor2img(translated)
            translated = (translated * 127.5 + 127.5).astype(np.uint8)
            
        # plot image
        if translated_only:
            plt.figure(figsize=(20, 20), dpi=100)
            plt.axis('off')
            plt.imshow(translated)

        else:
            # set title
            if ag:
                titles = ['Source', "Translated", "Attention mask", "Content Mask"]
            else:
                 titles = ['Source', "Translated"]
            _, ax = plt.subplots(1, len(titles), figsize=(20, 20))
            [ax[i].set_title(title) for i, title in enumerate(titles)]
            
            # show images
            if ag:
                translated_mask = util.tensor2img(translated_mask, isMask=True)
                translated_content = util.tensor2img(translated_content)
                translated_content = (translated_content * 127.5 + 127.5).astype(np.uint8)
                [ax[j].imshow(img) for j, img in enumerate([source_img, translated, translated_mask, translated_content])]
            else:
                [ax[j].imshow(img) for j, img in enumerate([source_img, translated])]
            
            [ax[j].axis("off") for j in range(len(titles))]

        # save
        plt.savefig(os.path.join(opt.out_dir, opt.name, fname[0]+'_all'), bbox_inches='tight', dpi=100)
        plt.close()