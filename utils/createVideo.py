import cv2
import os

image_folder = 'C:/Users\Juha/Documents/AI/datasets/aug-2023.tar/aug-2023/srv/data_fetching/road_camera_data/data/datalake/digitraffic/images/2023-08-23T10/C04507-vt3_Tampere_Lakalaiva/C0450702'
video_name = '../output/video/video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()