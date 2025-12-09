import os
import xml.etree.ElementTree as ET

classes = {
    "missing_hole": 0,
    "mouse_bite": 1,
    "open_circuit": 2,
    "short": 3,
    "spur": 4,
    "spurious_copper": 5
}

xml_root = "Annotations"
labels_folder = "labels"

os.makedirs(labels_folder, exist_ok=True)

# Find all XML files recursively
xml_files = []
for root_dir, _, files in os.walk(xml_root):
    for file in files:
        if file.endswith(".xml"):
            xml_files.append(os.path.join(root_dir, file))

print(f"Found {len(xml_files)} XML files. Converting...")

for xml_path in xml_files:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    w = int(root.find("size/width").text)
    h = int(root.find("size/height").text)

    txt_name = os.path.basename(xml_path).replace(".xml", ".txt")
    txt_path = os.path.join(labels_folder, txt_name)

    with open(txt_path, "w") as f:
        for obj in root.findall("object"):
            cls_name = obj.find("name").text.lower().strip()
            if cls_name not in classes:
                continue
            cls_id = classes[cls_name]

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            x_center = (xmin + xmax) / 2 / w
            y_center = (ymin + ymax) / 2 / h
            box_w = (xmax - xmin) / w
            box_h = (ymax - ymin) / h

            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

print("Conversion complete! Check your 'labels/' folder now.")
