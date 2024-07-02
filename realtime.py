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
clipFilesBySceneNumber = {}
currentClipPlaying = 'file:///C:/Users/NeN/Reconocimiento%20Emociones/clip/start.mp4'
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotionScore = {
	'Angry': 0, 'Disgust': 0, 'Fear':0, 'Happy':0, 'Sad':0, 'Surprise':0, 'Neutral':0
}
neutralEmotion = 'Neutral'
countClipAddedToPlayList = 0
clipAdded = False
neutralClipSelected = False
numScenes = 5
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
clipsChosen = []
emotionsByScene = {}

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
		return 'Clip: '+ self.path + ' E: '+ self.emotion + ' S: '+ str(self.score)
	def toGraphString(self):
		return str(self.numberScene)+'-'+self.path + ' E: '+ self.emotion + ' S: '+ str(self.score)

# Testea la asignación de emociones a la película. Asigna aleatoriamente el valor emocional que queremos darle a una escena
def testClipsChosen():
	for numberScene in range(numScenes):
		randomEmotion = selectRandomEmotion()
		randomScore = selectRandomScore()
		clip = Clip(numberScene,'clip/'+str(numberScene)+'.mp4',randomScore,randomEmotion,numberScene)
		clipsChosen.append(clip)

# Escoge una emoción de manera aleatoria    
def selectRandomEmotion():
	return random.choice(emotions)

# Escoge un valor aleatorio entre 2 y 5 para darselo a una escena
def selectRandomScore():
	return random.randrange(2,5)

# Añade por cada escena posible en la película y por cada emoción posible la puntuación de cada emoción de forma aleatoria para un clip de una escena
def addClips():
	for numberScene in range(numScenes):
		clipsInScene = []
		for numberEmotion in range(len(emotions)):
			currentEmotion = emotions[numberEmotion]
			clipFile = getFileName(numberScene, numberEmotion)
			player.media_new('clip/'+str(clipFile)+'.mp4')
			randomScore = selectRandomScore() # Se le asigna una puntuación a esa emoción (esto debería venir de BD del montador en este caso)
			randomScore = avoidingNeutralEmotion(currentEmotion,randomScore)
			clip = Clip(clipFile, 'clip/'+str(clipFile)+'.mp4', randomScore, currentEmotion, numberScene)
			print(clip.toString())
			clipsInScene.append(clip)
		clipFilesBySceneNumber[numberScene] = clipsInScene
	#input("Press Enter to continue...")
		
# Obtiene el nombre del fichero del clip de película en función del número de escenas posible y las emociones posibles (Esto vendría de BD)
def getFileName(numberScene, numberEmotion):
	return ((numberScene * len(emotions)) + numberEmotion) + 1 

# Intenta que las emociones Neutrales sean las últimas en escogerse aumentando el número de puntuación a la que tiene que llegar el usuario para que la emoción sea Neutral
def avoidingNeutralEmotion(currentEmotion, randomScore):
	if currentEmotion == 'Neutral':
		return randomScore * 10
	return randomScore
	
# Añade puntuación a la emoción concreta
def addScoreToEmotion(emotion,score):
	emotionScore[emotion] = emotionScore[emotion] + score

# Busca en el listado de clips los que contengan dicha emoción
def filterClipsByEmotion(clips, emotion):
	return list(filter(lambda m: m.emotion == emotion, clips))

# Busca en el listado de clips los que tengan un valor mayor a score
def filterClipsByScore(clips, score):
	return list(filter(lambda m: score >= m.score, clips))

# Devuelve de un listado de clips el que mayor score tenga
def findClipWithHighestScore(clips):
	clipSelected = clips[0]
	for clip in clips:
		if clip.score > clipSelected.score:
			clipSelected = clip
	return clipSelected

# Elige el siguiente clip a reproducir con mayor score, sino se escogió ninguno y se está llegando al final se elige el Neutral de esa escena
def chooseClipWithHighestScore():
	clipsByScore = {}
	clipSelected = False
	
	if countClipAddedToPlayList < numScenes:
		clipsByScene = list(clipFilesBySceneNumber[countClipAddedToPlayList])

		currentPosition = media_player.get_media_player().get_position()

		for emotion in emotions:
			clipsByEmotion = filterClipsByEmotion(clipsByScene, emotion)
			clipsByScore[emotion] = filterClipsByScore(clipsByEmotion, emotionScore[emotion])

		clipSelected = findClipByEmotionWithHighestScore(clipsByScore, clipSelected)

		clipSelected = checkClipAtEndOfPlayback(clipsByScene, currentPosition, clipSelected)

		#print('position at current clip ', currentPosition)

	return clipSelected

# Busca en una lista de clips filtrados por emociones los que tengan mayor valor
def findClipByEmotionWithHighestScore(clipsByScore, clipSelected):

	for emotion in emotions:
		if len(list(clipsByScore[emotion])) > 0:
			clipFromEmotionWithHighScore = findClipWithHighestScore(clipsByScore[emotion])
			if clipSelected == False:
				clipSelected = clipFromEmotionWithHighScore
			else:
				if clipSelected.score < clipFromEmotionWithHighScore.score:
					clipSelected = clipFromEmotionWithHighScore

	return clipSelected

# Si hemos llegado casi al final del clip y no se ha elegido un clip cogemos el Neutral
def checkClipAtEndOfPlayback(clipsByScene, currentPosition, clipSelected):
	global neutralClipSelected

	if currentPosition > 0.90 and clipSelected == False and len(clipsByScene) > 0 and neutralClipSelected == False:
		clipNeutralFromScene = filterClipsByEmotion(clipsByScene, neutralEmotion)
		clipSelected = clipNeutralFromScene[0]
		print('end of clip and no selected ', clipSelected.path)
		neutralClipSelected = True

	return clipSelected

# Limpia el array de puntuaciones para las emociones
def cleanEmotionScore():
	for emotion in emotions:
		emotionScore[emotion] = 0

# Actualiza el siguiente clip a reproducir si ha cambiado en la lista de reproduccion y limpia el del score anterior
def nextClip(actualClipPlaying):
	global countClipAddedToPlayList
	global currentClipPlaying
	global clipAdded
	global neutralClipSelected

	if actualClipPlaying != currentClipPlaying:
		countClipAddedToPlayList = countClipAddedToPlayList + 1
		currentClipPlaying = actualClipPlaying
		clipAdded = False
		neutralClipSelected = False		
		cleanEmotionScore()

# Añade las puntuaciones de las emociones de la escena actual
def addScoreEmotionToScene():
	emotionsByScene[countClipAddedToPlayList] = []
	for emotion in emotions:
		emotionsByScene[countClipAddedToPlayList].append(EmotionScore(emotion, emotionScore[emotion]))
		print(emotion + ' ',emotionScore[emotion])

"""
	Añade el clip con mayor puntuación o el Neutral en el caso de no encontrarlo a lista de reproducción.
	Pasa al siguiente clip en el caso de que la lista de reproducción haya pasado a reproduccir el siguiente.
"""
def checkClipToAddToPlayList():
	global countClipAddedToPlayList
	global currentClipPlaying
	global clipAdded

	clipSelected = chooseClipWithHighestScore()

	actualClipPlaying = media_player.get_media_player().get_media().get_mrl()

	if clipSelected != False and (clipAdded == False or neutralClipSelected == True):
		addScoreEmotionToScene()
		media_list.add_media(clipSelected.path)
		clipsChosen.append(clipSelected)
		clipAdded = True
		print('countClipAddedToPlayList ',countClipAddedToPlayList)
		print('actualClipPlaying ',actualClipPlaying)
		print('currentClipPlaying ', currentClipPlaying)
		print('clipAdded ', clipSelected.path)
		
	nextClip(actualClipPlaying)

# Inicializa la lista de reproducción y la pone a reproducir		
def startClip():
	media_list.add_media(mediaStart)
	# setting media list to the mediaplayer
	media_player.set_media_list(media_list)
	# start playing video
	media_player.play()

"""
	1. Captura lo que se graba en la webcam o videocamara del ordenador.
	2. Analiza de cada imagen los sentimientos del rostro del/o de los usuario/s y va actualizando el score por cada emoción que vaya/n sintiendo
"""
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
	# Se toma el tiempo para esperar a que se procesen las imagenes de las caras
	tic = time.time()
	# Comienza la captura de vídeo
	cap = cv2.VideoCapture(source)  # webcam
	# Mientras que el número de clips añadidos a la escena sea menor que el número de escenas se procesan las emociones de los usuarios en tiempo real
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

		# Se encuentra cara y los frames de la cara esta dentro del umbral para poder tratar la imagen y no se está procesando otra imagen
		if face_detected == True and face_included_frames == frame_threshold and freeze == False:
			freeze = True
			# base_img = img.copy()
			base_img = raw_img.copy()
			detected_faces_final = detected_faces.copy()
			tic = time.time()

		# Procesado de la imagen, se congela la imagen mientras se procesa
		if freeze == True:

			toc = time.time()
			# Si no he pasado el umbral de tiempo para procesar la imagen continuo procesando
			if (toc - tic) < time_threshold:

				if freezed_frame == 0:
					freeze_img = base_img.copy()

					for detected_face in detected_faces_final:
						x = detected_face[0]; y = detected_face[1]
						w = detected_face[2]; h = detected_face[3]
						
						# dibujo un cuadrado verde en la cara
						cv2.rectangle(freeze_img, (x, y), (x+w, y+h), (67, 67, 67),
						              1) 
						# -------------------------------

						# apply deep learning for custom_face

						custom_face = base_img[y:y+h, x:x+w]

						# -------------------------------
						# Analisis facial (emociones, edad, )
						if enable_face_analysis == True:
							# Devuelve la imagen en gris habiendola pasado por una capa convolucional
							gray_img = functions.preprocess_face(img=custom_face, target_size=(
							    48, 48), grayscale=True, enforce_detection=False, detector_backend=detector_backend)
							
							# -------------------------------
							# Analisis de emociones en tiempo real

							# Obtengo la predicción según el modelo de imagen convolucionada de cuales son las posibles emociones
							emotion_predictions = emotion_model.predict(gray_img)[0, :]
							# Obtengo el total de predicciones
							sum_of_predictions = emotion_predictions.sum()

							mood_items = []
							# Por cada una de las emociones obtengo el nombre para el label, el porcentaje que le corresponde según la predicción del modelo y lo meto en un array de mood (animo)
							for i in range(0, len(emotions)):
								mood_item = []
								emotion_label = emotions[i]
								emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
								mood_item.append(emotion_label)
								mood_item.append(emotion_prediction)
								mood_items.append(mood_item)

							# Con el array de emociones creo un DataFrame ordenado por score de la emoción que luego va ser pintado
							emotion_df = pd.DataFrame(mood_items, columns=["emotion", "score"])
							emotion_df = emotion_df.sort_values(
							    by=["score"], ascending=False).reset_index(drop=True)

							# background of mood box

							# transparency
							overlay = freeze_img.copy()
							opacity = 0.4
							dominant_emotion ='';

							# -------------------------------
							# Dibujo sobre la imagen el cuadrado para luego escribir las emociones
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
							# -------------------------------
							
							# Recorro el listado de emociones que ordenado por puntuación
							for index, instance in emotion_df.iterrows():
								emotion_label = "%s " % (instance['emotion'])
								emotion_score = instance['score']/100
								# Me quedo con la emoción dominante según el score
								if index == 0:
									dominant_emotion = "emotion: " + (instance['emotion']) +" percentage: "+ str(emotion_score)
								# Añado el score a mi listado de emociones
								addScoreToEmotion(instance['emotion'], emotion_score)
								# Compruebo si tengo que añadir un clip a la playlist según la puntuación añadida
								checkClipToAddToPlayList()

								bar_x = 35  # this is the size if an emotion is 100%
								bar_x = int(bar_x * emotion_score)

								# -------------------------------
								# Pintado del label de emociones en la pantalla
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

							# -------------------------------
										
							# -------------------------------
							# Analisis de edad en tiempo real

							face_224 = functions.preprocess_face(img = custom_face, target_size = (224, 224), grayscale = False, enforce_detection = False, detector_backend = detector_backend)

							age_predictions = age_model.predict(face_224)[0,:]
							apparent_age = Age.findApparentAge(age_predictions)

							# -------------------------------
							# Analisis de genero en tiempo real
							gender_prediction = gender_model.predict(face_224)[0,:]

							if np.argmax(gender_prediction) == 0:
								gender = "W"
							elif np.argmax(gender_prediction) == 1:
								gender = "M"

							print(str(int(apparent_age))," years old ", dominant_emotion, " ", gender)

							analysis_report = str(int(apparent_age))+" "+gender
							# -------------------------------
							
							# -------------------------------
							# Dibujo la edad y el genero en pantalla
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
				
				# -------------------------------				
				# Dibujo el tiempo restante para el siguiente análisis de una imagen capturada
				time_left = int(time_threshold - (toc - tic) + 1)

				cv2.rectangle(freeze_img, (10, 10), (90, 50), (67,67,67), -10)
				cv2.putText(freeze_img, str(time_left), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
				# -------------------------------

				cv2.imshow('img', freeze_img)

				freezed_frame = freezed_frame + 1
			# Ha pasado el umbral de tiempo para detectar y procesar las caras reiniciamos para poder empezar de nuevo
			else:
				face_detected = False
				face_included_frames = 0
				freeze = False
				freezed_frame = 0
		# He terminado de procesar la imagen añado el clip a la play list
		else:
			checkClipToAddToPlayList()
			cv2.imshow('img',img)

		if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
			break

	# kill open cv things
	cap.release()
	cv2.destroyAllWindows()

# POST PROCESS DECISIONS

# Comprueba si el clip pertenece a uno de los que ha sido elegido
def checkClipChosen(clip):
	for clipChosen in clipsChosen:
		if clip.number == clipChosen.number:
			return True 
	return False

# Muestra los resultados de las decisiones obtenidas durante el proceso de análisis de usuarios 
def getFinalDecisionChosen():
	result = ''
	for scene in range(numScenes):
		result+='\r\n------------ Scene: '+str(scene+1)+' ----------------- \r\n'
		result+='------------ Emotions Results ----------------- \r\n'
		for emotionResult in list(emotionsByScene.get(scene, [])):
			result+= emotionResult.toString() + '\r\n'
		result+='------------ Graph Results ----------------- \r\n'
		for clip in clipFilesBySceneNumber[scene]:
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

# Genera el QR con los clips que se haya escogido por los usuarios y por lo tanto de la rama resultante para poder compartir tu rama
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

# Dibuja el grafo de decisiones tomadas
def drawGraphFinalDecisionChosen():
	import networkx as nx
	import matplotlib.pyplot as plt
	from matplotlib.offsetbox import AnnotationBbox
	print("DRAW GRAPH DECISIONS")
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
		print("Scene: "+str(scene+1))
		G.add_node("Scene: "+str(scene+1))
		# Scene node position	
		node_position["Scene: "+str(scene+1)] = ( ((scene+1) * 10) + 10 , 46 + len(clipFilesBySceneNumber[scene])*10 )
		node_format["Scene: "+str(scene+1)] = {'color':'blue', 'size': 0}
			
		for index,clip in enumerate(clipFilesBySceneNumber[scene]):
			
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
	media_player.play()

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

#addClips()

#testClipsChosen()

#drawGraphFinalDecisionChosen()

addClips()

startClip()

analysis('', "Facenet", detector_backend='opencv', time_threshold=0.1, frame_threshold=1)

finalDecision = getFinalDecisionChosen()

drawGraphFinalDecisionChosen()

print(finalDecision)

input("Press Enter to continue...")