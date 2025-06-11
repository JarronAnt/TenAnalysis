from ultralytics import YOLO

model = YOLO('yolov8x')

res = model.predict('Inputs/Input_Vid.mp4', save=True)
print(res)
print("Bounding Boxes")
for Boxes in res[0].boxes:
    print(Boxes)