from ultralytics import YOLO

model = YOLO('models/best.pt')

res = model.predict('Inputs/Input_Vid.mp4', save=True)
print(res)
print("Bounding Boxes")
for Boxes in res[0].boxes:
    print(Boxes)