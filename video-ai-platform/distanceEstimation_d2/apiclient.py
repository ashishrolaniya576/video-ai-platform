import requests
import os
import json

API_URL = "http://localhost:8010/predict"   # change host/port if needed
IMAGE_PATH = "dataset/train/images/Screenshot from 2025-08-26 05-23-16.png"

def call_predict_api(image_path, fov=0.57, threshold=0.3, image_size=1920, padding=4):
    """
    Sends a POST request to the inference API.
    """
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/png")}
        data = {
            "fov": str(fov),
            "threshold": str(threshold),
            "image_size": str(image_size),
            "padding": str(padding),
            "return_images_b64": "0"   # set to "1" if you want images in base64
        }
        response = requests.post(API_URL, files=files, data=data)

    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.status_code, response.text)
        return None

if __name__ == "__main__":
    result = call_predict_api(IMAGE_PATH)

    if result:
        print(json.dumps(result, indent=2))

        # Download labeled image and chips if served via /files
        if "outputs" in result:
            print("\nDownloading labeled + chips...")
            out_dir = "downloads"
            os.makedirs(out_dir, exist_ok=True)

            labeled_name = result["outputs"]["labeled"]
            labeled_url = f"http://localhost:8010/files/{labeled_name}"
            r = requests.get(labeled_url)
            if r.status_code == 200:
                with open(os.path.join(out_dir, labeled_name), "wb") as f:
                    f.write(r.content)
                print("Saved labeled image:", labeled_name)

            for chip_name in result["outputs"]["chips"]:
                chip_url = f"http://localhost:8010/files/{chip_name}"
                r = requests.get(chip_url)
                if r.status_code == 200:
                    with open(os.path.join(out_dir, chip_name), "wb") as f:
                        f.write(r.content)
                    print("Saved chip:", chip_name)
