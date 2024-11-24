import json
from os import path as osp
import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

def is_valid_pose(pose_point):
    """
    유효한 포즈 좌표인지 확인.
    좌표 값이 (0, 0)이면 감지되지 않은 것으로 간주.
    """
    return not (pose_point[0] == 0.0 and pose_point[1] == 0.0)


def get_img_agnostic(img, parse, pose_data):
    """
    주어진 이미지(img), 분할 이미지(parse), 포즈 데이터(pose_data)를 바탕으로
    특정 부위를 마스킹한 결과 이미지 생성.
    """
    # 1. Parsing mask 생성
    parse_array = np.array(parse)
    parse_head = ((parse_array == 4).astype(np.float32) +
                  (parse_array == 13).astype(np.float32))
    parse_lower = ((parse_array == 9).astype(np.float32) +
                   (parse_array == 12).astype(np.float32) +
                   (parse_array == 16).astype(np.float32) +
                   (parse_array == 17).astype(np.float32) +
                   (parse_array == 18).astype(np.float32) +
                   (parse_array == 19).astype(np.float32))

    # 2. 마스킹 작업을 위한 초기화
    agnostic = img.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    # 3. 팔 길이를 기준으로 마스킹 크기 계산
    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    r = int(length_a / 16) + 1

    # 4. 하체 길이 계산 및 유효성 검사
    if is_valid_pose(pose_data[9]) and is_valid_pose(pose_data[12]):
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a
    else:
        length_b = 0  # 하체가 감지되지 않은 경우

    # 5. 팔 마스킹
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*10)
    for i in [2, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
    
    # 팔의 세부 마스킹 복구 (팔꿈치와 손목)
    for i in [3, 4, 6, 7]:
        if (not is_valid_pose(pose_data[i - 1])) or (not is_valid_pose(pose_data[i])):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

    # 6. 상체와 목 마스킹
    if length_b > 0:  # 하체가 감지된 경우에만 처리
        for i in [9, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

    # 목 마스킹
    if is_valid_pose(pose_data[1]):
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'gray', 'gray')

    # 7. 머리와 하의 복원
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

    return agnostic


if __name__ =="__main__":
    data_path = './HR-VITON/test/test'
    output_path = './HR-VITON/test/test/agnostic-v3.2'
    
    os.makedirs(output_path, exist_ok=True)
    
    for im_name in tqdm(os.listdir(osp.join(data_path, 'image'))):
        
        # load pose image
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        
        try:
            with open(osp.join(data_path, 'openpose_json', pose_name), 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['people'][0]['pose_keypoints_2d']
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 3))[:, :2]
        except IndexError:
            print(pose_name)
            continue

        # load parsing image
        im = Image.open(osp.join(data_path, 'image', im_name))
        label_name = im_name.replace('.jpg', '.png')
        im_label = Image.open(osp.join(data_path, 'image-parse-v3', label_name))

        agnostic = get_img_agnostic(im, im_label, pose_data)
        
        agnostic.save(osp.join(output_path, im_name))