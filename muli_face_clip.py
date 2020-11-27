'''
包含人脸的图片分割出人脸的批处理脚本
命令行输入 以下语句，即可分割出人脸并保存在faces文件夹：
python muli_face_clip.py --img_dir liqin
如果报错：可能是图片中人脸不清晰，程序无法识别，删掉报错图片，重新运行即可。
'''
import cv2
import dlib
import os
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer by Pytorch')
    parser.add_argument('--img_dir', type=str, default='imgs',
                        help='style images directory')
    parser.add_argument('--save_dir', type=str, default='faces',
                        help='save directory for result and loss')
    args = parser.parse_args()

    # create directory to save
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)


    for filename in os.listdir(args.img_dir):
        print(filename)
        predictor_model = 'shape_predictor_68_face_landmarks.dat'
        detector = dlib.get_frontal_face_detector()  # dlib人脸检测器，获得人脸框位置的检测器
        predictor = dlib.shape_predictor(predictor_model)  # 获得人脸关键点检测器

        # cv2读取图像
        img = cv2.imread(args.img_dir+'/'+filename)  # 读进来直接是BGR 格式数据格式在 0~255
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2.cvtColor(p1,p2) 是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。
                # cv2.COLOR_BGR2RGB 将BGR格式转换成RGB格式；   cv2.COLOR_BGR2GRAY 将BGR格式转换成灰度图片
        # 人脸数rects
        rects = detector(img, 0)  # 返回的是人脸的 bounding box
        # faces存储full_object_detection对象
        faces = dlib.full_object_detections()

        for i in range(len(rects)):
            faces.append(predictor(img, rects[i]))

        face_images = dlib.get_face_chips(img, faces, size=512)
        for image in face_images:
            cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{args.save_dir}/{filename}', cv_bgr_img)


if __name__ == '__main__':
    main()