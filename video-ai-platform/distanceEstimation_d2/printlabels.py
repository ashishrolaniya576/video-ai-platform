import os
import xml.etree.ElementTree as ET
import pandas as pd

def parse_annotations(xml_dir):
    records = []

    # Iterate over all .xml files in the directory
    for file in os.listdir(xml_dir):
        if file.endswith(".xml"):
            file_path = os.path.join(xml_dir, file)
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()

                filename = root.find("filename").text if root.find("filename") is not None else file

                # Extract all <object> details
                for obj in root.findall("object"):
                    label = obj.find("name").text.strip() if obj.find("name") is not None else "Unknown"
                    fov = obj.find("fov").text.strip() if obj.find("fov") is not None else "N/A"
                    distance = obj.find("distance").text.strip() if obj.find("distance") is not None else "N/A"

                    records.append({
                        "Image": filename,
                        "Label": label,
                        "FOV": fov,
                        "Distance": distance
                    })

            except Exception as e:
                print(f"Error parsing {file}: {e}")

    return records


if __name__ == "__main__":
    xml_directory = "dataset/valid/labels/"  # change this path
    output_csv = "valid_annotations_summary.csv"

    data = parse_annotations(xml_directory)

    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)

        print(f"✅ CSV file created: {output_csv}")
        print("\n🔹 Preview of extracted data:")
        print(df.head())

        # Print training data summary
        print("\n📊 Training Data Summary (Label Counts):")
        summary = df["Label"].value_counts()
        print(summary)

    else:
        print("No data found.")
