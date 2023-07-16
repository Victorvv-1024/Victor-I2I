from os import listdir
import os
import re
import matplotlib.pyplot as plt

if __name__ == '__main__':

    path = ['evaluation/victor_dataset/test_tar',
            'evaluation/victor_dataset/ag/ag_vic_lr1e-6_pix1/translated',
            'evaluation/victor_dataset/ag/ag_vic_lr1e-6_pix2/translated',
            'evaluation/victor_dataset/wag/vic_lossG/translated'
            ]
    
    cols = [listdir(x) for x in path]

    sort_cols = [sorted(col,
                        key = lambda fname: int(re.split('[=  .]', fname)[1])) for col in cols]
    
    combine_list = list(zip(*sort_cols))

    save_path = 'evaluation/victor_dataset/compare'

    titles = ['source', 'AG pix1', 'AG pix2', 'WAG']

    for i in range(0, len(combine_list), 10):
        # if i == 400:
        #     nrow = 9
        # else: nrow = 10
        nrow = 10
        fig, axes = plt.subplots(nrow, len(titles), figsize=(20, 20))
        [axes[0, i].set_title(title, fontsize = 30) for i, title in enumerate(titles)]

        fig.subplots_adjust(hspace=0.0, wspace=0.0, top=0.95, bottom=0.05, left=0.17, right=0.845)

        for ax, imgname in zip(axes, combine_list[i:i+10]):
            img = [plt.imread(os.path.join(path[j], imgname[j]))for j in range(len(imgname))]
            # print(img)
            [ax[j].imshow(img) for j, img in enumerate(img)]
            [ax[j].axis("off") for j in range(len(img))]
        fig.savefig(f'{save_path}/{i}.pdf', bbox_inches='tight', dpi=100)
        print('done')
        plt.close()