import pandas as pd
import qrcode
import os

data = pd.read_csv("qr_urls_dataset.csv")

os.makedirs("dataset/phishing", exist_ok=True)
os.makedirs("dataset/legitimate", exist_ok=True)

for i, row in data.iterrows():

    url = str(row["URL"])
    label = row["label"]

    # Skip URLs that are too long for QR encoding
    if len(url) > 1000:
        continue

    try:
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4
        )

        qr.add_data(url)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")

        if label == 1:
            img.save(f"dataset/phishing/qr_{i}.png")
        else:
            img.save(f"dataset/legitimate/qr_{i}.png")

    except Exception:
        continue

print("QR dataset generated successfully")