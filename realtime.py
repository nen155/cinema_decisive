from deepface.detectors import FaceDetector
from deepface.commons import functions, distance as dst
from deepface.extendedmodels import Age
from deepface import DeepFace
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import time
# importing vlc module
import vlc
import random

# creating Instance class object
player = vlc.Instance()
# creating a new media list
media_list = player.media_list_new()
# creating a media player object
media_player = player.media_list_player_new()
# creating a new media
mediaStart = player.media_new('clip/start.mp4')
clip_files = {}
currentClipPlaying = 'file:///C:/Users/NeN/Reconocimiento%20Emociones/clip/start.mp4'
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotionScore = {
	'Angry': 0, 'Disgust': 0, 'Fear':0, 'Happy':0, 'Sad':0, 'Surprise':0, 'Neutral':0
}
countClipAddedToPlayList = 0
clipAdded = False
baseClipSelected = False
numScenes = 5
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
clipsChosen = []
emotionsByScene = {}
models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
]

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe'
]

class EmotionScore:
	def __init__(self, emotion, score):
		self.emotion = emotion
		self.score = score
	def toString(self):
		return 'Emotion: '+ self.emotion + ' Score: '+ str(self.score)

class Clip:
	def __init__(self, number, path, score, emotion, numberScene):
		self.number = number
		self.path = path
		self.score = score
		self.emotion = emotion
		self.numberScene = numberScene
	def toString(self):
		return 'Clip: '+ self.path + '\r\nEmotion: '+ self.emotion + ' Score: '+ str(self.score) + '\r\nScene: '+ str(self.numberScene)
	def toShortString(self):
		return 'Clip: '+ self.path + ' Emotion: '+ self.emotion + ' Score: '+ str(self.score) + ' Scene: '+ str(self.numberScene)
	def toGraphString(self):
		return str(self.numberScene)+'-'+self.path + ' E: '+ self.emotion + ' S: '+ str(self.score)

def testClipsChosen():
	for numberScene in range(numScenes):
		randomEmotion = selectRandomEmotion()
		randomScore = selectRandomScore()
		clip = Clip(numberScene,'clip/'+str(numberScene)+'.mp4',randomScore,randomEmotion,numberScene)
		clipsChosen.append(clip)

    
def selectRandomEmotion():
	return random.choice(emotions)

def selectRandomScore():
	return random.randrange(2,5)

def addClips():
	for numberScene in range(numScenes):
		clipsInScene = []
		for numberEmotion in range((len(emotions) - 1)):
			clip_file = ((numberScene * (len(emotions) - 1)) + numberEmotion) + 1
			player.media_new('clip/'+str(clip_file)+'.mp4')
			randomEmotion = selectRandomEmotion()
			randomScore = selectRandomScore()
			if randomEmotion == 'Neutral':
				randomScore = randomScore * 3
			clip = Clip(clip_file,'clip/'+str(clip_file)+'.mp4',randomScore,randomEmotion,numberScene)
			print(clip.toString())
			clipsInScene.append(clip)
		clip_files[numberScene] = clipsInScene
	#input("Press Enter to continue...")

def ponderingEmotion(emotion,score):
	emotionScore[emotion] = emotionScore[emotion] + score

def filterClipsByEmotion(clips, emotion):
	return list(filter(lambda m: m.emotion == emotion, clips))

def filterClipsByScore(clips, score):
	return list(filter(lambda m: score >= m.score, clips))

def chooseClipPondered():
	global clipAdded
	global baseClipSelected
	clipsByScore = {}
	clipSelected = False
	
	if countClipAddedToPlayList < numScenes:
		clipsByScene = list(clip_files[countClipAddedToPlayList])

		for emotion in emotions:
			clipsByScore[emotion] = filterClipsByScore(filterClipsByEmotion(clipsByScene, emotion), emotionScore[emotion])

		for emotion in emotions:
			if len(list(clipsByScore[emotion])) > 0:
				clipSelected = clipsByScore[emotion][0]

		current_position = media_player.get_media_player().get_position()

		# We are almost at the end of clip and no selected next clip, so select one at random
		if current_position > 0.90 and clipSelected == False and len(clipsByScene) > 0 and baseClipSelected == False:
			clipSelected = clipsByScene[0]
			print('end of clip and no selected ', clipSelected.path)
			baseClipSelected = True

	return clipSelected

def cleanEmotionScore():
	for emotion in emotions:
		emotionScore[emotion] = 0

def nextClip(actualClipPlaying):
	global countClipAddedToPlayList
	global currentClipPlaying
	global clipAdded
	global baseClipSelected

	if actualClipPlaying != currentClipPlaying:
		countClipAddedToPlayList = countClipAddedToPlayList + 1
		currentClipPlaying = actualClipPlaying
		clipAdded = False
		baseClipSelected = False		
		cleanEmotionScore()

def addScoreEmotionToScene():
	emotionsByScene[countClipAddedToPlayList] = []
	for emotion in emotions:
		emotionsByScene[countClipAddedToPlayList].append(EmotionScore(emotion, emotionScore[emotion]))
		print(emotion + ' ',emotionScore[emotion])

def addClipToPlayList():
	global countClipAddedToPlayList
	global currentClipPlaying
	global clipAdded

	clipSelected = chooseClipPondered()

	actualClipPlaying = media_player.get_media_player().get_media().get_mrl()

	if clipSelected != False and (clipAdded == False or baseClipSelected == True):
		addScoreEmotionToScene()
		media_list.add_media(clipSelected.path)
		clipsChosen.append(clipSelected)
		clipAdded = True
		print('countClipAddedToPlayList ',countClipAddedToPlayList)
		print('actualClipPlaying ',actualClipPlaying)
		print('currentClipPlaying ', currentClipPlaying)
		print('clipAdded ', clipSelected.path)
		
	nextClip(actualClipPlaying)
		
def startClip():
	media_list.add_media(mediaStart)
	# setting media list to the mediaplayer
	media_player.set_media_list(media_list)
	# start playing video
	media_player.play()

def analysis(db_path, model_name='VGG-Face', detector_backend='opencv', distance_metric='cosine', enable_face_analysis=True, source=0, time_threshold=5, frame_threshold=5):
	# ------------------------

	face_detector = FaceDetector.build_model(detector_backend)
	print("Detector backend is ", detector_backend)

	# ------------------------

	input_shape = (
	    224, 224); input_shape_x = input_shape[0]; input_shape_y = input_shape[1]

	text_color = (255, 255, 255)

	employees = []
	# check passed db folder exists
	if os.path.isdir(db_path) == True:
		for r, d, f in os.walk(db_path):  # r=root, d=directories, f = files
			for file in f:
				if ('.jpg' in file):
					# exact_path = os.path.join(r, file)
					exact_path = r + "/" + file
					# print(exact_path)
					employees.append(exact_path)

	if len(employees) == 0:
		print("WARNING: There is no image in this path ( ", db_path,
		      ") . Face recognition will not be performed.")

	# ------------------------

	if len(employees) > 0:

		model = DeepFace.build_model(model_name)
		print(model_name, " is built")

		# ------------------------

		input_shape = functions.find_input_shape(model)
		input_shape_x = input_shape[0]; input_shape_y = input_shape[1]

		# tuned thresholds for model and metric pair
		threshold = dst.findThreshold(model_name, distance_metric)

	# ------------------------
	# facial attribute analysis models

	if enable_face_analysis == True:

		tic = time.time()

		emotion_model = DeepFace.build_model('Emotion')
		print("Emotion model loaded")

		age_model = DeepFace.build_model('Age')
		print("Age model loaded")

		gender_model = DeepFace.build_model('Gender')
		print("Gender model loaded")

		toc = time.time()

		print("Facial attibute analysis models loaded in ", toc-tic, " seconds")

	# ------------------------

	# find embeddings for employee list

	tic = time.time()

	# -----------------------

	pbar = tqdm(range(0, len(employees)), desc='Finding embeddings')

	# TODO: why don't you store those embeddings in a pickle file similar to find function?

	embeddings = []
	# for employee in employees:
	for index in pbar:
		employee = employees[index]
		pbar.set_description("Finding embedding for %s" % (employee.split("/")[-1]))
		embedding = []

		# preprocess_face returns single face. this is expected for source images in db.
		img = functions.preprocess_face(img=employee, target_size=(
		    input_shape_y, input_shape_x), enforce_detection=False, detector_backend=detector_backend)
		img_representation = model.predict(img)[0, :]

		embedding.append(employee)
		embedding.append(img_representation)
		embeddings.append(embedding)

	df = pd.DataFrame(embeddings, columns=['employee', 'embedding'])
	df['distance_metric'] = distance_metric

	toc = time.time()

	print("Embeddings found for given data set in ", toc-tic, " seconds")

	# -----------------------

	pivot_img_size = 112  # face recognition result image

	# -----------------------

	freeze = False
	face_detected = False
	face_included_frames = 0  # freeze screen if face detected sequantially 5 frames
	freezed_frame = 0
	tic = time.time()

	cap = cv2.VideoCapture(source)  # webcam

	while(countClipAddedToPlayList < numScenes):
		ret, img = cap.read()

		if img is None:
			break

		raw_img = img.copy()
		resolution_x = img.shape[1]; resolution_y = img.shape[0]

		if freeze == False:

			try:
				# faces store list of detected_face and region pair
				faces = FaceDetector.detect_faces(
				    face_detector, detector_backend, img, align=False)
			except:  # to avoid exception if no face detected
				faces = []

			if len(faces) == 0:
				face_included_frames = 0
		else:
			faces = []

		detected_faces = []
		face_index = 0
		for face, (x, y, w, h) in faces:
			if w > 130:  # discard small detected faces

				face_detected = True
				if face_index == 0:
					face_included_frames = face_included_frames + \
					    1  # increase frame for a single face

				cv2.rectangle(img, (x, y), (x+w, y+h), (67, 67, 67),
				              1)  # draw rectangle to main image

				cv2.putText(img, str(frame_threshold - face_included_frames), (int(x+w/4),
				            int(y+h/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)

				detected_face = img[int(y):int(y+h), int(x):int(x+w)]  # crop detected face

				# -------------------------------------

				detected_faces.append((x, y, w, h))
				face_index = face_index + 1

				# -------------------------------------

		if face_detected == True and face_included_frames == frame_threshold and freeze == False:
			freeze = True
			# base_img = img.copy()
			base_img = raw_img.copy()
			detected_faces_final = detected_faces.copy()
			tic = time.time()

		if freeze == True:

			toc = time.time()
			if (toc - tic) < time_threshold:

				if freezed_frame == 0:
					freeze_img = base_img.copy()

					for detected_face in detected_faces_final:
						x = detected_face[0]; y = detected_face[1]
						w = detected_face[2]; h = detected_face[3]

						cv2.rectangle(freeze_img, (x, y), (x+w, y+h), (67, 67, 67),
						              1)  # draw rectangle to main image

						# -------------------------------

						# apply deep learning for custom_face

						custom_face = base_img[y:y+h, x:x+w]

						# -------------------------------
						# facial attribute analysis

						if enable_face_analysis == True:

							gray_img = functions.preprocess_face(img=custom_face, target_size=(
							    48, 48), grayscale=True, enforce_detection=False, detector_backend='opencv')
							emotion_labels = ['Angry', 'Disgust', 'Fear',
							    'Happy', 'Sad', 'Surprise', 'Neutral']
							emotion_predictions = emotion_model.predict(gray_img)[0, :]
							sum_of_predictions = emotion_predictions.sum()

							mood_items = []
							for i in range(0, len(emotion_labels)):
								mood_item = []
								emotion_label = emotion_labels[i]
								emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
								mood_item.append(emotion_label)
								mood_item.append(emotion_prediction)
								mood_items.append(mood_item)

							emotion_df = pd.DataFrame(mood_items, columns=["emotion", "score"])
							emotion_df = emotion_df.sort_values(
							    by=["score"], ascending=False).reset_index(drop=True)

							# background of mood box

							# transparency
							overlay = freeze_img.copy()
							opacity = 0.4
							dominant_emotion ='';

							if x+w+pivot_img_size < resolution_x:
								# right
								cv2.rectangle(freeze_img									# , (x+w,y+20)
									, (x+w, y)									, (x+w+pivot_img_size, y+h)									, (64, 64, 64), cv2.FILLED)

								cv2.addWeighted(overlay, opacity, freeze_img,
								                1 - opacity, 0, freeze_img)

							elif x-pivot_img_size > 0:
								# left
								cv2.rectangle(freeze_img									# , (x-pivot_img_size,y+20)
									, (x-pivot_img_size, y)									, (x, y+h)									, (64, 64, 64), cv2.FILLED)

								cv2.addWeighted(overlay, opacity, freeze_img,
								                1 - opacity, 0, freeze_img)

							for index, instance in emotion_df.iterrows():
								emotion_label = "%s " % (instance['emotion'])
								emotion_score = instance['score']/100
								
								if index == 0:
									dominant_emotion = "emotion: " + (instance['emotion']) +" percentage: "+ str(emotion_score)

								ponderingEmotion(instance['emotion'], emotion_score)

								addClipToPlayList()

								bar_x = 35  # this is the size if an emotion is 100%
								bar_x = int(bar_x * emotion_score)

								if x+w+pivot_img_size < resolution_x:

									text_location_y = y + 20 + (index+1) * 20
									text_location_x = x+w

									if text_location_y < y + h:
										cv2.putText(freeze_img, emotion_label, (text_location_x, text_location_y),
										            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

										cv2.rectangle(freeze_img											, (x+w+70, y + 13 + (index+1) * 20)											,
										              (x+w+70+bar_x, y + 13 + (index+1) * 20 + 5)											, (255, 255, 255), cv2.FILLED)

								elif x-pivot_img_size > 0:

									text_location_y = y + 20 + (index+1) * 20
									text_location_x = x-pivot_img_size

									if text_location_y <= y+h:
										cv2.putText(freeze_img, emotion_label, (text_location_x, text_location_y),
										            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

										cv2.rectangle(freeze_img											, (x-pivot_img_size+70, y + 13 + (index+1) * 20)											,
										              (x-pivot_img_size+70+bar_x, y + 13 + (index+1) * 20 + 5)											, (255, 255, 255), cv2.FILLED)

							# -------------------------------

							face_224 = functions.preprocess_face(img = custom_face, target_size = (224, 224), grayscale = False, enforce_detection = False, detector_backend = 'opencv')

							age_predictions = age_model.predict(face_224)[0,:]
							apparent_age = Age.findApparentAge(age_predictions)

							# -------------------------------

							gender_prediction = gender_model.predict(face_224)[0,:]

							if np.argmax(gender_prediction) == 0:
								gender = "W"
							elif np.argmax(gender_prediction) == 1:
								gender = "M"

							print(str(int(apparent_age))," years old ", dominant_emotion, " ", gender)

							analysis_report = str(int(apparent_age))+" "+gender

							# -------------------------------

							info_box_color = (46,200,255)

							# top
							if y - pivot_img_size + int(pivot_img_size/5) > 0:

								triangle_coordinates = np.array( [
									(x+int(w/2), y)
									, (x+int(w/2)-int(w/10), y-int(pivot_img_size/3))
									, (x+int(w/2)+int(w/10), y-int(pivot_img_size/3))
								] )

								cv2.drawContours(freeze_img, [triangle_coordinates], 0, info_box_color, -1)

								cv2.rectangle(freeze_img, (x+int(w/5), y-pivot_img_size+int(pivot_img_size/5)), (x+w-int(w/5), y-int(pivot_img_size/3)), info_box_color, cv2.FILLED)

								cv2.putText(freeze_img, analysis_report, (x+int(w/3.5), y - int(pivot_img_size/2.1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)

							# bottom
							elif y + h + pivot_img_size - int(pivot_img_size/5) < resolution_y:

								triangle_coordinates = np.array( [
									(x+int(w/2), y+h)
									, (x+int(w/2)-int(w/10), y+h+int(pivot_img_size/3))
									, (x+int(w/2)+int(w/10), y+h+int(pivot_img_size/3))
								] )

								cv2.drawContours(freeze_img, [triangle_coordinates], 0, info_box_color, -1)

								cv2.rectangle(freeze_img, (x+int(w/5), y + h + int(pivot_img_size/3)), (x+w-int(w/5), y+h+pivot_img_size-int(pivot_img_size/5)), info_box_color, cv2.FILLED)

								cv2.putText(freeze_img, analysis_report, (x+int(w/3.5), y + h + int(pivot_img_size/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)

						# -------------------------------
						# #face recognition

						# custom_face = functions.preprocess_face(img = custom_face, target_size = (input_shape_y, input_shape_x), enforce_detection = False, detector_backend = 'opencv')

						# #check preprocess_face function handled
						# if custom_face.shape[1:3] == input_shape:
						# 	if df.shape[0] > 0: #if there are images to verify, apply face recognition
						# 		img1_representation = model.predict(custom_face)[0,:]

						# 		#print(freezed_frame," - ",img1_representation[0:5])

						# 		def findDistance(row):
						# 			distance_metric = row['distance_metric']
						# 			img2_representation = row['embedding']

						# 			distance = 1000 #initialize very large value
						# 			if distance_metric == 'cosine':
						# 				distance = dst.findCosineDistance(img1_representation, img2_representation)
						# 			elif distance_metric == 'euclidean':
						# 				distance = dst.findEuclideanDistance(img1_representation, img2_representation)
						# 			elif distance_metric == 'euclidean_l2':
						# 				distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))

						# 			return distance

						# 		df['distance'] = df.apply(findDistance, axis = 1)
						# 		df = df.sort_values(by = ["distance"])

						# 		candidate = df.iloc[0]
						# 		employee_name = candidate['employee']
						# 		best_distance = candidate['distance']

						# 		#print(candidate[['employee', 'distance']].values)

						# 		#if True:
						# 		if best_distance <= threshold:
						# 			#print(employee_name)
						# 			display_img = cv2.imread(employee_name)

						# 			display_img = cv2.resize(display_img, (pivot_img_size, pivot_img_size))

						# 			label = employee_name.split("/")[-1].replace(".jpg", "")
						# 			label = re.sub('[0-9]', '', label)

						# 			try:
						# 				if y - pivot_img_size > 0 and x + w + pivot_img_size < resolution_x:
						# 					#top right
						# 					freeze_img[y - pivot_img_size:y, x+w:x+w+pivot_img_size] = display_img

						# 					overlay = freeze_img.copy(); opacity = 0.4
						# 					cv2.rectangle(freeze_img,(x+w,y),(x+w+pivot_img_size, y+20),(46,200,255),cv2.FILLED)
						# 					cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

						# 					cv2.putText(freeze_img, label, (x+w, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

						# 					#connect face and text
						# 					cv2.line(freeze_img,(x+int(w/2), y), (x+3*int(w/4), y-int(pivot_img_size/2)),(67,67,67),1)
						# 					cv2.line(freeze_img, (x+3*int(w/4), y-int(pivot_img_size/2)), (x+w, y - int(pivot_img_size/2)), (67,67,67),1)

						# 				elif y + h + pivot_img_size < resolution_y and x - pivot_img_size > 0:
						# 					#bottom left
						# 					freeze_img[y+h:y+h+pivot_img_size, x-pivot_img_size:x] = display_img

						# 					overlay = freeze_img.copy(); opacity = 0.4
						# 					cv2.rectangle(freeze_img,(x-pivot_img_size,y+h-20),(x, y+h),(46,200,255),cv2.FILLED)
						# 					cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

						# 					cv2.putText(freeze_img, label, (x - pivot_img_size, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

						# 					#connect face and text
						# 					cv2.line(freeze_img,(x+int(w/2), y+h), (x+int(w/2)-int(w/4), y+h+int(pivot_img_size/2)),(67,67,67),1)
						# 					cv2.line(freeze_img, (x+int(w/2)-int(w/4), y+h+int(pivot_img_size/2)), (x, y+h+int(pivot_img_size/2)), (67,67,67),1)

						# 				elif y - pivot_img_size > 0 and x - pivot_img_size > 0:
						# 					#top left
						# 					freeze_img[y-pivot_img_size:y, x-pivot_img_size:x] = display_img

						# 					overlay = freeze_img.copy(); opacity = 0.4
						# 					cv2.rectangle(freeze_img,(x- pivot_img_size,y),(x, y+20),(46,200,255),cv2.FILLED)
						# 					cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

						# 					cv2.putText(freeze_img, label, (x - pivot_img_size, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

						# 					#connect face and text
						# 					cv2.line(freeze_img,(x+int(w/2), y), (x+int(w/2)-int(w/4), y-int(pivot_img_size/2)),(67,67,67),1)
						# 					cv2.line(freeze_img, (x+int(w/2)-int(w/4), y-int(pivot_img_size/2)), (x, y - int(pivot_img_size/2)), (67,67,67),1)

						# 				elif x+w+pivot_img_size < resolution_x and y + h + pivot_img_size < resolution_y:
						# 					#bottom righ
						# 					freeze_img[y+h:y+h+pivot_img_size, x+w:x+w+pivot_img_size] = display_img

						# 					overlay = freeze_img.copy(); opacity = 0.4
						# 					cv2.rectangle(freeze_img,(x+w,y+h-20),(x+w+pivot_img_size, y+h),(46,200,255),cv2.FILLED)
						# 					cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

						# 					cv2.putText(freeze_img, label, (x+w, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

						# 					#connect face and text
						# 					cv2.line(freeze_img,(x+int(w/2), y+h), (x+int(w/2)+int(w/4), y+h+int(pivot_img_size/2)),(67,67,67),1)
						# 					cv2.line(freeze_img, (x+int(w/2)+int(w/4), y+h+int(pivot_img_size/2)), (x+w, y+h+int(pivot_img_size/2)), (67,67,67),1)
						# 			except Exception as err:
						# 				print(str(err))

						# tic = time.time() #in this way, freezed image can show 5 seconds

						# #-------------------------------

				time_left = int(time_threshold - (toc - tic) + 1)

				cv2.rectangle(freeze_img, (10, 10), (90, 50), (67,67,67), -10)
				cv2.putText(freeze_img, str(time_left), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

				cv2.imshow('img', freeze_img)

				freezed_frame = freezed_frame + 1
			else:
				face_detected = False
				face_included_frames = 0
				freeze = False
				freezed_frame = 0

		else:
			addClipToPlayList()
			cv2.imshow('img',img)

		if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
			break

	# kill open cv things
	cap.release()
	cv2.destroyAllWindows()

#Post Process decisions

def checkClipChosen(clip):
	for clipChosen in clipsChosen:
		if clip.number == clipChosen.number:
			return True 
	return False

def getFinalDecisionChosen():
	result = ''
	for scene in range(numScenes):
		result+='\r\n------------ Scene: '+str(scene)+' ----------------- \r\n'
		result+='------------ Emotions Results ----------------- \r\n'
		for emotionResult in list(emotionsByScene.get(scene, [])):
			result+= emotionResult.toString() + '\r\n'
		result+='------------ Graph Results ----------------- \r\n'
		for clip in clip_files[scene]:
			if checkClipChosen(clip):
				result += '**Chosen: ' + clip.toShortString()
			else:
				result += clip.toShortString()
			result += '\t'
		result += '\r\n'
	return result

def getImage(path):
   from matplotlib.offsetbox import OffsetImage
   import matplotlib.pyplot as plt
   return OffsetImage(plt.imread(path, format="png"), zoom=.07)

def generateQrCode(data):
	# Importing library
	import qrcode
	
	qr = qrcode.QRCode(version = 1,
                   box_size = 50,
                   border = 5)

	# Encoding data using make() function
	qr.add_data(data)
 
	qr.make(fit = True)
	img = qr.make_image(fill_color = 'black',
						back_color = 'white')
	img.save("qrcode.png")

def drawGraphFinalDecisionChosen():
	import networkx as nx
	import matplotlib.pyplot as plt
	from matplotlib.offsetbox import AnnotationBbox
	# Crear un grafo vacío
	G = nx.Graph()
	clipsHasBeenChosen = []
	objectsClipsChosen = []
	node_format = {'start.mp4': {'color':'red', 'size': 300} }
	node_position = {'start.mp4': (13,75)}

	plt.rcParams["figure.figsize"] = [7.50, 3.50]
	plt.rcParams["figure.autolayout"] = True
	fig, ax = plt.subplots()

	# Agregar nodos al grafo
	lastClipChosen = 'start.mp4'
	G.add_node('start.mp4')
	for scene in range(numScenes):

		G.add_node("Scene: "+str(scene+1))
		# Scene node position	
		node_position["Scene: "+str(scene+1)] = ( ((scene+1) * 10) + 10 , 46 + len(clip_files[scene])*10 )
		node_format["Scene: "+str(scene+1)] = {'color':'blue', 'size': 0}
			
		for index,clip in enumerate(clip_files[scene]):
			
			node_position[clip.toGraphString()] = ( ((scene+1) * 10) + 10 , (index*10) + 50 )

			ab = AnnotationBbox(getImage("film.png"), ( ((scene+1) * 10) + 10 , (index*10) + 50 ), frameon=False)
			ax.add_artist(ab)
			
			if checkClipChosen(clip):
				if len(clipsHasBeenChosen) > 0:
					lastClipChosen = clipsHasBeenChosen[len(clipsHasBeenChosen) -1]
				G.add_node(clip.toGraphString())
				G.add_edge(lastClipChosen, clip.toGraphString())
				clipsHasBeenChosen.append(clip.toGraphString())
				objectsClipsChosen.append(clip)
				node_format[clip.toGraphString()] = {'color':'red', 'size': 300}
			else:
				G.add_node(clip.toGraphString())
				node_format[clip.toGraphString()] = {'color':'blue', 'size': 300}
	
	# Generar QR code
	listIdsClips = '-'.join(list(map(lambda x : str(x.number), objectsClipsChosen)))
	generateQrCode(listIdsClips)

	# Qr code node graph
	G.add_node("Share your decisions!")
	ab = AnnotationBbox(getImage("qrcode.png"), ((numScenes*10)+20, 50), frameon=False)
	ax.add_artist(ab)
	node_position["Share your decisions!"] = ((numScenes*10)+20, 55)
	node_format["Share your decisions!"] = {'color':'red', 'size': 0}

	# Dibujar el grafo
	nx.draw(G, node_position, with_labels=True, node_shape = "o", arrows = True, font_size=8, node_color=[node_format[node]['color'] for node in G.nodes()], node_size=[node_format[node]['size'] for node in G.nodes()])
	#plt.show()

	fig = plt.gcf()
	fig.set_size_inches((15, 10), forward=False)
	plt.savefig("node-graph-decisions.png", dpi=100)
	graph = player.media_new("node-graph-decisions.png")
	media_list.add_media(graph)
	media_player.set_media_list(media_list)
	media_player.play()

# addClips()

# testClipsChosen()

# drawGraphFinalDecisionChosen()

addClips()

startClip()

analysis('',models[1], detector_backend=backends[3],time_threshold=0.1,frame_threshold=1)

finalDecision = getFinalDecisionChosen()

drawGraphFinalDecisionChosen()

print(finalDecision)

input("Press Enter to continue...")