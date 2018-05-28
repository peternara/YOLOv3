import xml.dom.minidom as dom
import numpy as np


def xml_extractor(dir):
    DOMTree = dom.parse(dir)                                        # Parse the [Annotation xml] file of PASCAL VOC2007
    collection = DOMTree.documentElement                            # get xml file root
    file_name_xml = collection.getElementsByTagName("filename")[0]  # get TAG [filename] from root , use [0] cause maybe more than one file
    objects_xml = collection.getElementsByTagName("object")         # get TAG [object] from root
    size_xml = collection.getElementsByTagName("size")              # get TAG [size] from root

    file_name = file_name_xml.childNodes[0].data

    width, height = "", ""
    for size in size_xml:                                           # get width and height from TAG size
        width = size.getElementsByTagName("width")[0]
        height = size.getElementsByTagName("height")[0]

        width = width.childNodes[0].data
        height = height.childNodes[0].data

    objects = []
    for object_xml in objects_xml:                                  # get bndbox,xmin,ymin,xmax,ymax from TAG object
        object_name = object_xml.getElementsByTagName("name")[0]
        bdbox = object_xml.getElementsByTagName("bndbox")[0]
        xmin = bdbox.getElementsByTagName("xmin")[0]
        ymin = bdbox.getElementsByTagName("ymin")[0]
        xmax = bdbox.getElementsByTagName("xmax")[0]
        ymax = bdbox.getElementsByTagName("ymax")[0]

        object = ( object_name.childNodes[0].data,
                   xmin.childNodes[0].data,
                   ymin.childNodes[0].data,
                   xmax.childNodes[0].data,
                   ymax.childNodes[0].data)
        objects.append(object)

    return file_name, width, height, objects


# print(xml_extractor("/home/sherman/projects/data/VOCdevkit/VOC2007/Annotations/009961.xml"))


def labels_normalizer(batches_filenames, target_width, target_height, layerout_width, layerout_height):
    class_map = {
        'person': 5,
        'bird': 6,
        'cat': 7,
        'cow': 8,
        'dog': 9,
        'horse': 10,
        'sheep': 11,
        'aeroplane': 12,
        'bicycle': 13,
        'boat': 14,
        'bus': 15,
        'car': 16,
        'motorbike': 17,
        'train': 18,
        'bottle': 19,
        'chair': 20,
        'diningtable': 21,
        'pottedplant': 22,
        'sofa': 23,
        'tvmonitor': 24,
    }

    height_width = []
    batches_labels = []

    for bfs in batches_filenames:
        bls = []
        for filename in bfs:
            _, width, height, objects = xml_extractor(filename)
            width_proprotion = target_width / int(width)
            height_proprotion = target_height / int(height)
            label = np.add(np.zeros([int(layerout_height), int(layerout_width), 255]), 1e-8)
            for obj in objects:
                class_label = class_map[obj[0]]
                xmin = float(obj[1])
                ymin = float(obj[2])
                xmax = float(obj[3])
                ymax = float(obj[4])
                x = ((1.0 * xmax + xmin) / 2) * width_proprotion          # 计算目标中点的x值
                y = ((1.0 * ymax + xmin) / 2) * height_proprotion         # 计算目标中点的y值
                bdbox_width = (1.0 * xmax - xmin) * width_proprotion    # 计算目标的boundding box的宽
                bdbox_height = (1.0 * ymax - ymin) * height_proprotion  # 计算目标的boundding box的高
                flag_width = int(target_width) / layerout_width         # 计算一个box内含有多少个原图像的横轴像素
                flag_height = int(target_height) / layerout_height      # 计算一个box内含有多少个原图像的横轴像素
                box_x = x // flag_width     # 计算x所属的box的x下标  整除
                box_y = y // flag_height    # 计算y所属的box的y下标  整除
                if box_x == layerout_width:             # 把最后一个box右边界的点归为最后一个box管理（本来为下一个box管理）
                    box_x -= 1
                if box_y == layerout_height:            # 把最下面一个box的下边界的点归为最下面一个box管理（本来为下一个box管理）
                    box_y -= 1
                for i in range(3):      # 每个box预测3个bdbox
                    label[int(box_y), int(box_x), i * 25] = x                       # point x
                    label[int(box_y), int(box_x), i * 25 + 1] = y                   # point y
                    label[int(box_y), int(box_x), i * 25 + 2] = bdbox_width         # bdbox width
                    label[int(box_y), int(box_x), i * 25 + 3] = bdbox_height        # bdbox height
                    label[int(box_y), int(box_x), i * 25 + 4] = 1                   # objectness
                    label[int(box_y), int(box_x), i * 25 + int(class_label)] = 0.9  # class label
                print(label.shape)
            bls.append(label)
        batches_labels.append(bls)
    return batches_labels


'''--------Test extract_labels--------'''
if __name__ == '__main__':
    dir = [['/home/sherman/projects/data/VOCdevkit/VOC2007/Annotations/009961.xml',
            '/home/sherman/projects/data/VOCdevkit/VOC2007/Annotations/009959.xml'],
           ['/home/sherman/projects/data/VOCdevkit/VOC2007/Annotations/000033.xml',
            '/home/sherman/projects/data/VOCdevkit/VOC2007/Annotations/009950.xml']]
    batches_labels = labels_normalizer(dir, 512, 512, 16, 16)
    print(np.array(dir).shape)
    print(np.array(batches_labels).shape)


