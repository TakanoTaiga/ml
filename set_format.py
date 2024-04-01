from PIL import Image
import os

source_dir = './input_image'
dest_dir = './out_image'

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

files = os.listdir(source_dir)

counter = 1

for file in files:
    file_path = os.path.join(source_dir, file)
    if ".folder_ml" in file_path:
        continue
    try:
        with Image.open(file_path) as img:
            img_resized = img.resize((640, 360))
            new_file_name = f'image-{counter:02}.jpg'
            new_file_path = os.path.join(dest_dir, new_file_name)
            img_resized.save(new_file_path)
            print(f'{file} to {new_file_name}')
            counter += 1
    except IOError:
        print(f'{file} is not image file')

print('finished')

