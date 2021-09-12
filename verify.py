from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import dlib
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import face_recognition



def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def is_blurry(frame, threshold = 100.0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    text = False
    if fm < threshold:
        text = True
    return text

def get_box(image, detector):
    (h, w) = image.shape[:2]
    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)), 1.0, (300, 300),
    (104.0, 177.0, 123.0), swapRB=False, crop=False)
    
    # apply OpenCV's deep learning-based face detector to localize faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()
    confidence = detections[0,0,:,2]
    most_confident = np.argmax(confidence)
    
    box = detections[0, 0, most_confident, 3:7] * np.array([w, h, w, h])
    confidence = detections[0, 0, most_confident, 2]
    
    return box, confidence

def get_face(frame):
    frame = imutils.resize(frame, width=600)
    
    box, confidence = get_box(frame, detector)
    (startX, startY, endX, endY) = box.astype("int")
    
    dlibrect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_aligned = fa.align(frame, gray, dlibrect)
    
    return face_aligned, confidence




# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, help="path to image")
ap.add_argument("-o", "--output", type=str, help="path to optional output video file")
ap.add_argument("-c", "--threshold", type=float, default=0.5, help="minimum threshold to compare embeddings")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = "models/deploy.prototxt"
modelPath = "models/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256)


print("[INFO] grab and process the face on ID card...")
image = cv2.imread(args["input"])
(h, w) = image.shape[:2]
# construct a blob from the image
imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
	(104.0, 177.0, 123.0), swapRB=False, crop=False)
# localize the face
detector.setInput(imageBlob)
detections = detector.forward()
# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with
	# the prediction
	confidence = detections[0, 0, i, 2]
	# filter out weak detections
	if confidence > 0.9:
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		# face alignment
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		dlibrect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
		face_aligned = fa.align(image, gray, dlibrect)
		face_aligned = face_aligned[32:224,32:224]
		(fH, fW) = face_aligned.shape[:2]
		# ensure the face width and height are sufficiently large
		if fW < 20 or fH < 20:
			continue
		# encoding
		try:
			face_encoding_ID = face_recognition.face_encodings(face_aligned)[0]
		except:
			continue
face_ref = face_aligned
image_ref = imutils.resize(image, width=192)


print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# start the FPS throughput estimator
fps = FPS().start()

# loop over frames from the video file stream
while True:

	# grab the frame from the threaded video stream
	frame = vs.read()
	# if is_blurry(frame):
	# 	continue

	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	# frame = imutils.resize(frame, width=800)
	(h, w) = frame.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > 0.7:
			# compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# face alignment
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			dlibrect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
			face_aligned = fa.align(frame, gray, dlibrect)
			# face_aligned = face_aligned[64:192,64:192]
			face_aligned = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2RGB)
			face_aligned = face_aligned[32:224,32:224]
			(fH, fW) = face_aligned.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# encoding
			try:
				face_encoding = face_recognition.face_encodings(face_aligned)[0]
			except:
				continue

			# use the known face with the smallest distance to the new face
			distance = face_recognition.face_distance([face_encoding_ID], face_encoding)

			# draw the bounding box of the face along with the associated distance
			if distance < args["threshold"]:
				# text = f'{name}_{round(distance,1)}'
				text = 'IDENTITAS TERKONFIRMASI'
				color = (0,255,0)
			else:
				text = 'IDENTITAS TIDAK TERKONFIRMASI'
				color = (0,0,255)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

			# put the name
			font_scale = 1
			font = cv2.FONT_ITALIC
			(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
			# set the text start position
			text_offset_x, text_offset_y = w//2, 40
			# make the coords of the box with a small padding of two pixels
			box_coords = ((text_offset_x - 5, text_offset_y + 5), (text_offset_x + text_width + 5, text_offset_y - text_height - 5))
			overlay = frame.copy()
			cv2.rectangle(overlay, box_coords[0], box_coords[1], color, -1)
			cv2.putText(overlay, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(255,255,255), thickness=2)
			# apply the overlay
			alpha=0.6
			cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

	# combine current frame with image_ref
	frame[:192,:192] = face_ref
	(hi, wi) = image_ref.shape[:2]
	frame[192:192+hi,:192] = image_ref

	# if the frame dimensions are empty, set them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# if we are supposed to be writing a video to disk, initialize the writer
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

	# update the FPS counter
	fps.update()

	# check to see if we should write the frame to disk
	if writer is not None:
		writer.write(frame)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()