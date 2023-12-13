import cv2
import os

image_folder = 'C:/Users\Juha\Documents\AI\datasets/traffic_camera_images\oct-2023\srv\data_fetching/road_camera_data\data\datalake\digitraffic\images/2023-09-29T10/full_video'
video_name = '../output/video/video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()