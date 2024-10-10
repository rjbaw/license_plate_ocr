import os

save_dir = './label-deploy/'
for fi in os.listdir(save_dir):
    imgname, suffix = os.path.splitext(fi)
    label = imgname.split('_')[0]
    i = 1
    save_path = save_dir + label + '_' + str(i) + '.jpg'

    while os.path.exists(save_path):
        i += 1
        save_path = save_dir + label + '_' + str(i) + '.jpg'

    old_path = save_dir + fi
    os.rename(old_path, save_path)



