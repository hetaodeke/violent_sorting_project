import os
import json

def extract(input_path, output_path):
    video_dir = 'data/ava/videos/'
    video_name_list = os.listdir(video_dir)
    with open(input_path, 'r+') as ipf:
        for line in ipf.readlines():
            video_name = line.split(',')[0]
            if video_name in video_name_list:
                with open(output_path, 'a') as opf:
                    opf.write(line)

def extract_from_file(input_path, output_path):
    file_path = 'data/ava/frame_list/val_extract.csv'
    video_name_list = []
    with open(file_path, 'r+') as f:
        for line in f.readlines():
            video_name = line.split(' ')[0]
            video_name_list.append(video_name)
    video_list = list(set(video_name_list))
    with open(input_path, 'r+') as ipf:
        for line in ipf.readlines():
            video_name = line.split(',')[0]
            if video_name in video_list:
                with open(output_path, 'a') as opf:
                    opf.write(line)

def imgpath_modification(input_path,output_path):
    with open(input_path, 'r+') as f:
        for line in f.readlines()[1:]:
            line_info = line.split(' ')
            address_lst = line_info[3].split('/')
            frame_lst = address_lst[1].rsplit('_', 1)
            frame_lst[0] = 'img'
            frame = '_'.join(frame_lst)
            address_lst[1] = frame
            address = '/'.join(address_lst)
            line_info[3] = address
            new_line = ' '.join(line_info) 
            with open(output_path, 'a') as of:
                of.write(new_line)

def get_classname_json(input_path, output_path):
    with open(input_path, 'r+') as f:
        i = 1
        class_id = []
        class_name = []
        for line in f.readlines():
            if i %2 == 0:
                class_id.append(int(line.rstrip('\n').split(':')[1]))
            else:
                class_name.append(line.rstrip('\n').split(':')[1])
            i += 1
        class_dict = dict(zip(class_name, class_id))
    with open(output_path, 'r+') as of:
        json.dump(class_dict, of)
        


if __name__ == '__main__':
    data_dir = 'data/ava/annotations/ava_action_list_v2.1_for_activitynet_2018_copy.pbtxt'
    output_path = 'data/ava/annotations/ava_classnames.json'
    get_classname_json(data_dir, output_path)