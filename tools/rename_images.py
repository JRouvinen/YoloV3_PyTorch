import os

def rename_images(folder_path, start_number):
    count = start_number
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            new_name = f"{count}.jpg"  # Change the extension if needed
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
            count += 1

# Usage example
folder_path = "C:/Users\Juha\Documents\AI\datasets/traffic_camera_images\oct-2023\srv\data_fetching/road_camera_data\data\datalake\digitraffic\images/2023-09-29T10/to_video_3"
folder_path = folder_path.replace("\"", "/")
folder_path = folder_path.replace("\\", "/")
# Replace with the actual folder path
start_number = 288
# Replace with the desired starting number
rename_images(folder_path, start_number)