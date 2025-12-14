import base64

with open("face2.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

with open("face2_base64.txt", "w") as f:
    f.write(b64)

print("Base64 saved to face_base64.txt")
