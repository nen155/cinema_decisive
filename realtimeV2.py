# project dependencies
from deepface import DeepFace
from deepface.commons import logger as log

# built-in dependencies
import os
import time
from typing import List, Tuple, Optional

from tqdm import tqdm
# 3rd party dependencies
import numpy as np
import pandas as pd
import cv2

# importing vlc module
import vlc
import random

logger = log.get_singletonish_logger()

# dependency configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

IDENTIFIED_IMG_SIZE = 112
TEXT_COLOR = (255, 255, 255)


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
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
emotionScore = {
	'angry': 0, 'disgust': 0, 'fear':0, 'happy':0, 'sad':0, 'surprise':0, 'neutral':0
}
neutralEmotion = 'neutral'
countClipAddedToPlayList = 0
clipAdded = False
neutralClipSelected = False
numScenes = 5
clipsChosen = []
emotionsByScene = {}

###################################### FUNCIONES PROCESADO ########################################################

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
		return randomScore * 3
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

###################################### FIN FUNCIONES PROCESADO ########################################################
	
###################################### FUNCIONES POST PROCESADO ########################################################

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

###################################### FIN FUNCIONES POST PROCESADO ########################################################

###################################### ANALISIS ###################################################

def build_facial_recognition_model(model_name: str) -> None:
	"""
	Build facial recognition model
	Args:
		model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
			OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).
	Returns
		input_shape (tuple): input shape of given facial recognitio n model.
	"""
	_ = DeepFace.build_model(model_name=model_name)
	logger.info(f"{model_name} is built")

def search_identity(
	detected_face: np.ndarray,
	db_path: str,
	model_name: str,
	detector_backend: str,
	distance_metric: str,
) -> Tuple[Optional[str], Optional[np.ndarray]]:
	"""
	Search an identity in facial database.
	Args:
		detected_face (np.ndarray): extracted individual facial image
		db_path (string): Path to the folder containing image files. All detected faces
			in the database will be considered in the decision-making process.
		model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
			OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).
		detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
			'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
			(default is opencv).
		distance_metric (string): Metric for measuring similarity. Options: 'cosine',
			'euclidean', 'euclidean_l2' (default is cosine).
	Returns:
		result (tuple): result consisting of following objects
			identified image path (str)
			identified image itself (np.ndarray)
	"""
	target_path = None
	try:
		dfs = DeepFace.find(
			img_path=detected_face,
			db_path=db_path,
			model_name=model_name,
			detector_backend=detector_backend,
			distance_metric=distance_metric,
			enforce_detection=False,
			silent=True,
		)
	except ValueError as err:
		if f"No item found in {db_path}" in str(err):
			logger.warn(
				f"No item is found in {db_path}."
				"So, no facial recognition analysis will be performed."
			)
			dfs = []
		else:
			raise err
	if len(dfs) == 0:
		# you may consider to return unknown person's image here
		return None, None

	# detected face is coming from parent, safe to access 1st index
	df = dfs[0]

	if df.shape[0] == 0:
		return None, None

	candidate = df.iloc[0]
	target_path = candidate["identity"]
	logger.info(f"Hello, {target_path}")

	# load found identity image - extracted if possible
	target_objs = DeepFace.extract_faces(
		img_path=target_path,
		detector_backend=detector_backend,
		enforce_detection=False,
		align=True,
	)

	# extract facial area of the identified image if and only if it has one face
	# otherwise, show image as is
	if len(target_objs) == 1:
		# extract 1st item directly
		target_obj = target_objs[0]
		target_img = target_obj["face"]
		target_img = cv2.resize(target_img, (IDENTIFIED_IMG_SIZE, IDENTIFIED_IMG_SIZE))
		target_img *= 255
		target_img = target_img[:, :, ::-1]
	else:
		target_img = cv2.imread(target_path)

	return target_path.split("/")[-1], target_img

def build_demography_models(enable_face_analysis: bool) -> None:
	"""
	Build demography analysis models
	Args:
		enable_face_analysis (bool): Flag to enable face analysis (default is True).
	Returns:
		None
	"""
	if enable_face_analysis is False:
		return
	DeepFace.build_model(model_name="Age")
	logger.info("Age model is just built")
	DeepFace.build_model(model_name="Gender")
	logger.info("Gender model is just built")
	DeepFace.build_model(model_name="Emotion")
	logger.info("Emotion model is just built")

def highlight_facial_areas(
	img: np.ndarray,
	faces_coordinates: List[Tuple[int, int, int, int, bool, float]],
	anti_spoofing: bool = False,
) -> np.ndarray:
	"""
	Highlight detected faces with rectangles in the given image
	Args:
		img (np.ndarray): image itself
		faces_coordinates (list): list of face coordinates as tuple with x, y, w and h
			also is_real and antispoof_score keys
		anti_spoofing (boolean): Flag to enable anti spoofing (default is False).
	Returns:
		img (np.ndarray): image with highlighted facial areas
	"""
	for x, y, w, h, is_real, antispoof_score in faces_coordinates:
		# highlight facial area with rectangle

		if anti_spoofing is False:
			color = (67, 67, 67)
		else:
			if is_real is True:
				color = (0, 255, 0)
			else:
				color = (0, 0, 255)
		cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
	return img

def countdown_to_freeze(
	img: np.ndarray,
	faces_coordinates: List[Tuple[int, int, int, int, bool, float]],
	frame_threshold: int,
	num_frames_with_faces: int,
) -> np.ndarray:
	"""
	Highlight time to freeze in the image's facial areas
	Args:
		img (np.ndarray): image itself
		faces_coordinates (list): list of face coordinates as tuple with x, y, w and h
		frame_threshold (int): how many sequantial frames required with face(s) to freeze
		num_frames_with_faces (int): how many sequantial frames do we have with face(s)
	Returns:
		img (np.ndarray): image with counter values
	"""
	for x, y, w, h, is_real, antispoof_score in faces_coordinates:
		cv2.putText(
			img,
			str(frame_threshold - (num_frames_with_faces % frame_threshold)),
			(int(x + w / 4), int(y + h / 1.5)),
			cv2.FONT_HERSHEY_SIMPLEX,
			4,
			(255, 255, 255),
			2,
		)
	return img

def countdown_to_release(
	img: Optional[np.ndarray], tic: float, time_threshold: int
) -> Optional[np.ndarray]:
	"""
	Highlight time to release the freezing in the image top left area
	Args:
		img (np.ndarray): image itself
		tic (float): time specifying when freezing started
		time_threshold (int): freeze time threshold
	Returns:
		img (np.ndarray): image with time to release the freezing
	"""
	# do not take any action if it is not frozen yet
	if img is None:
		return img
	toc = time.time()
	time_left = int(time_threshold - (toc - tic) + 1)
	cv2.rectangle(img, (10, 10), (90, 50), (67, 67, 67), -10)
	cv2.putText(
		img,
		str(time_left),
		(40, 40),
		cv2.FONT_HERSHEY_SIMPLEX,
		1,
		(255, 255, 255),
		1,
	)
	return img

def grab_facial_areas(
	img: np.ndarray, detector_backend: str, threshold: int = 130, anti_spoofing: bool = False
) -> List[Tuple[int, int, int, int, bool, float]]:
	"""
	Find facial area coordinates in the given image
	Args:
		img (np.ndarray): image itself
		detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
			'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
			(default is opencv).
		threshold (int): threshold for facial area, discard smaller ones
	Returns
		result (list): list of tuple with x, y, w and h coordinates
	"""
	try:
		face_objs = DeepFace.extract_faces(
			img_path=img,
			detector_backend=detector_backend,
			# you may consider to extract with larger expanding value
			expand_percentage=0,
			anti_spoofing=anti_spoofing,
		)
		faces = [
			(
				face_obj["facial_area"]["x"],
				face_obj["facial_area"]["y"],
				face_obj["facial_area"]["w"],
				face_obj["facial_area"]["h"],
				face_obj.get("is_real", True),
				face_obj.get("antispoof_score", 0),
			)
			for face_obj in face_objs
			if face_obj["facial_area"]["w"] > threshold
		]
		return faces
	except:  # to avoid exception if no face detected
		return []

def extract_facial_areas(
	img: np.ndarray, faces_coordinates: List[Tuple[int, int, int, int, bool, float]]
) -> List[np.ndarray]:
	"""
	Extract facial areas as numpy array from given image
	Args:
		img (np.ndarray): image itself
		faces_coordinates (list): list of facial area coordinates as tuple with
			x, y, w and h values also is_real and antispoof_score keys
	Returns:
		detected_faces (list): list of detected facial area images
	"""
	detected_faces = []
	for x, y, w, h, is_real, antispoof_score in faces_coordinates:
		detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]
		detected_faces.append(detected_face)
	return detected_faces

def perform_facial_recognition(
	img: np.ndarray,
	detected_faces: List[np.ndarray],
	faces_coordinates: List[Tuple[int, int, int, int, bool, float]],
	db_path: str,
	detector_backend: str,
	distance_metric: str,
	model_name: str,
) -> np.ndarray:
	"""
	Perform facial recognition
	Args:
		img (np.ndarray): image itself
		detected_faces (list): list of extracted detected face images as numpy
		faces_coordinates (list): list of facial area coordinates as tuple with
			x, y, w and h values also is_real and antispoof_score keys
		db_path (string): Path to the folder containing image files. All detected faces
			in the database will be considered in the decision-making process.
		detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
			'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
			(default is opencv).
		distance_metric (string): Metric for measuring similarity. Options: 'cosine',
			'euclidean', 'euclidean_l2' (default is cosine).
		model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
			OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).
	Returns:
		img (np.ndarray): image with identified face informations
	"""
	for idx, (x, y, w, h, is_real, antispoof_score) in enumerate(faces_coordinates):
		detected_face = detected_faces[idx]
		target_label, target_img = search_identity(
			detected_face=detected_face,
			db_path=db_path,
			detector_backend=detector_backend,
			distance_metric=distance_metric,
			model_name=model_name,
		)
		if target_label is None:
			continue

		img = overlay_identified_face(
			img=img,
			target_img=target_img,
			label=target_label,
			x=x,
			y=y,
			w=w,
			h=h,
		)

	return img

def perform_demography_analysis(
	enable_face_analysis: bool,
	img: np.ndarray,
	faces_coordinates: List[Tuple[int, int, int, int, bool, float]],
	detected_faces: List[np.ndarray],
) -> np.ndarray:
	"""
	Perform demography analysis on given image
	Args:
		enable_face_analysis (bool): Flag to enable face analysis.
		img (np.ndarray): image itself
		faces_coordinates (list): list of face coordinates as tuple with
			x, y, w and h values also is_real and antispoof_score keys
		detected_faces (list): list of extracted detected face images as numpy
	Returns:
		img (np.ndarray): image with analyzed demography information
	"""
	if enable_face_analysis is False:
		return img
	for idx, (x, y, w, h, is_real, antispoof_score) in enumerate(faces_coordinates):
		detected_face = detected_faces[idx]
		demographies = DeepFace.analyze(
			img_path=detected_face,
			actions=("age", "gender", "emotion"),
			detector_backend="skip",
			enforce_detection=False,
			silent=True,
		)

		if len(demographies) == 0:
			continue

		# safe to access 1st index because detector backend is skip
		demography = demographies[0]

		img = overlay_emotion(img=img, emotion_probas=demography["emotion"], x=x, y=y, w=w, h=h)
		img = overlay_age_gender(
			img=img,
			apparent_age=demography["age"],
			gender=demography["dominant_gender"][0:1],  # M or W
			x=x,
			y=y,
			w=w,
			h=h,
		)
	return img

def overlay_identified_face(
	img: np.ndarray,
	target_img: np.ndarray,
	label: str,
	x: int,
	y: int,
	w: int,
	h: int,
) -> np.ndarray:
	"""
	Overlay the identified face onto image itself
	Args:
		img (np.ndarray): image itself
		target_img (np.ndarray): identified face's image
		label (str): name of the identified face
		x (int): x coordinate of the face on the given image
		y (int): y coordinate of the face on the given image
		w (int): w coordinate of the face on the given image
		h (int): h coordinate of the face on the given image
	Returns:
		img (np.ndarray): image with overlayed identity
	"""
	try:
		if y - IDENTIFIED_IMG_SIZE > 0 and x + w + IDENTIFIED_IMG_SIZE < img.shape[1]:
			# top right
			img[
				y - IDENTIFIED_IMG_SIZE : y,
				x + w : x + w + IDENTIFIED_IMG_SIZE,
			] = target_img

			overlay = img.copy()
			opacity = 0.4
			cv2.rectangle(
				img,
				(x + w, y),
				(x + w + IDENTIFIED_IMG_SIZE, y + 20),
				(46, 200, 255),
				cv2.FILLED,
			)
			cv2.addWeighted(
				overlay,
				opacity,
				img,
				1 - opacity,
				0,
				img,
			)

			cv2.putText(
				img,
				label,
				(x + w, y + 10),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5,
				TEXT_COLOR,
				1,
			)

			# connect face and text
			cv2.line(
				img,
				(x + int(w / 2), y),
				(x + 3 * int(w / 4), y - int(IDENTIFIED_IMG_SIZE / 2)),
				(67, 67, 67),
				1,
			)
			cv2.line(
				img,
				(x + 3 * int(w / 4), y - int(IDENTIFIED_IMG_SIZE / 2)),
				(x + w, y - int(IDENTIFIED_IMG_SIZE / 2)),
				(67, 67, 67),
				1,
			)

		elif y + h + IDENTIFIED_IMG_SIZE < img.shape[0] and x - IDENTIFIED_IMG_SIZE > 0:
			# bottom left
			img[
				y + h : y + h + IDENTIFIED_IMG_SIZE,
				x - IDENTIFIED_IMG_SIZE : x,
			] = target_img

			overlay = img.copy()
			opacity = 0.4
			cv2.rectangle(
				img,
				(x - IDENTIFIED_IMG_SIZE, y + h - 20),
				(x, y + h),
				(46, 200, 255),
				cv2.FILLED,
			)
			cv2.addWeighted(
				overlay,
				opacity,
				img,
				1 - opacity,
				0,
				img,
			)

			cv2.putText(
				img,
				label,
				(x - IDENTIFIED_IMG_SIZE, y + h - 10),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5,
				TEXT_COLOR,
				1,
			)

			# connect face and text
			cv2.line(
				img,
				(x + int(w / 2), y + h),
				(
					x + int(w / 2) - int(w / 4),
					y + h + int(IDENTIFIED_IMG_SIZE / 2),
				),
				(67, 67, 67),
				1,
			)
			cv2.line(
				img,
				(
					x + int(w / 2) - int(w / 4),
					y + h + int(IDENTIFIED_IMG_SIZE / 2),
				),
				(x, y + h + int(IDENTIFIED_IMG_SIZE / 2)),
				(67, 67, 67),
				1,
			)

		elif y - IDENTIFIED_IMG_SIZE > 0 and x - IDENTIFIED_IMG_SIZE > 0:
			# top left
			img[y - IDENTIFIED_IMG_SIZE : y, x - IDENTIFIED_IMG_SIZE : x] = target_img

			overlay = img.copy()
			opacity = 0.4
			cv2.rectangle(
				img,
				(x - IDENTIFIED_IMG_SIZE, y),
				(x, y + 20),
				(46, 200, 255),
				cv2.FILLED,
			)
			cv2.addWeighted(
				overlay,
				opacity,
				img,
				1 - opacity,
				0,
				img,
			)

			cv2.putText(
				img,
				label,
				(x - IDENTIFIED_IMG_SIZE, y + 10),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5,
				TEXT_COLOR,
				1,
			)

			# connect face and text
			cv2.line(
				img,
				(x + int(w / 2), y),
				(
					x + int(w / 2) - int(w / 4),
					y - int(IDENTIFIED_IMG_SIZE / 2),
				),
				(67, 67, 67),
				1,
			)
			cv2.line(
				img,
				(
					x + int(w / 2) - int(w / 4),
					y - int(IDENTIFIED_IMG_SIZE / 2),
				),
				(x, y - int(IDENTIFIED_IMG_SIZE / 2)),
				(67, 67, 67),
				1,
			)

		elif (
			x + w + IDENTIFIED_IMG_SIZE < img.shape[1]
			and y + h + IDENTIFIED_IMG_SIZE < img.shape[0]
		):
			# bottom righ
			img[
				y + h : y + h + IDENTIFIED_IMG_SIZE,
				x + w : x + w + IDENTIFIED_IMG_SIZE,
			] = target_img

			overlay = img.copy()
			opacity = 0.4
			cv2.rectangle(
				img,
				(x + w, y + h - 20),
				(x + w + IDENTIFIED_IMG_SIZE, y + h),
				(46, 200, 255),
				cv2.FILLED,
			)
			cv2.addWeighted(
				overlay,
				opacity,
				img,
				1 - opacity,
				0,
				img,
			)

			cv2.putText(
				img,
				label,
				(x + w, y + h - 10),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5,
				TEXT_COLOR,
				1,
			)

			# connect face and text
			cv2.line(
				img,
				(x + int(w / 2), y + h),
				(
					x + int(w / 2) + int(w / 4),
					y + h + int(IDENTIFIED_IMG_SIZE / 2),
				),
				(67, 67, 67),
				1,
			)
			cv2.line(
				img,
				(
					x + int(w / 2) + int(w / 4),
					y + h + int(IDENTIFIED_IMG_SIZE / 2),
				),
				(x + w, y + h + int(IDENTIFIED_IMG_SIZE / 2)),
				(67, 67, 67),
				1,
			)
		else:
			logger.info("cannot put facial recognition info on the image")
	except Exception as err:  # pylint: disable=broad-except
		logger.error(str(err))
	return img

def overlay_emotion(
	img: np.ndarray, emotion_probas: dict, x: int, y: int, w: int, h: int
) -> np.ndarray:
	"""
	Overlay the analyzed emotion of face onto image itself
	Args:
		img (np.ndarray): image itself
		emotion_probas (dict): probability of different emotionas dictionary
		x (int): x coordinate of the face on the given image
		y (int): y coordinate of the face on the given image
		w (int): w coordinate of the face on the given image
		h (int): h coordinate of the face on the given image
	Returns:
		img (np.ndarray): image with overlay emotion analsis results
	"""
	emotion_df = pd.DataFrame(emotion_probas.items(), columns=["emotion", "score"])
	emotion_df = emotion_df.sort_values(by=["score"], ascending=False).reset_index(drop=True)

	# background of mood box

	# transparency
	overlay = img.copy()
	opacity = 0.4

	# put gray background to the right of the detected image
	if x + w + IDENTIFIED_IMG_SIZE < img.shape[1]:
		cv2.rectangle(
			img,
			(x + w, y),
			(x + w + IDENTIFIED_IMG_SIZE, y + h),
			(64, 64, 64),
			cv2.FILLED,
		)
		cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
	
	# put gray background to the left of the detected image
	elif x - IDENTIFIED_IMG_SIZE > 0:
		cv2.rectangle(
			img,
			(x - IDENTIFIED_IMG_SIZE, y),
			(x, y + h),
			(64, 64, 64),
			cv2.FILLED,
		)
		cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

	for index, instance in emotion_df.iterrows():
		current_emotion = instance["emotion"]
		emotion_label = f"{current_emotion} "
		emotion_score = instance["score"] / 100

		addScoreToEmotion(instance['emotion'], emotion_score)

		checkClipToAddToPlayList()
		
		filled_bar_x = 35  # this is the size if an emotion is 100%
		bar_x = int(filled_bar_x * emotion_score)

		if x + w + IDENTIFIED_IMG_SIZE < img.shape[1]:

			text_location_y = y + 20 + (index + 1) * 20
			text_location_x = x + w

			if text_location_y < y + h:
				cv2.putText(
					img,
					emotion_label,
					(text_location_x, text_location_y),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.5,
					(255, 255, 255),
					1,
				)

				cv2.rectangle(
					img,
					(x + w + 70, y + 13 + (index + 1) * 20),
					(
						x + w + 70 + bar_x,
						y + 13 + (index + 1) * 20 + 5,
					),
					(255, 255, 255),
					cv2.FILLED,
				)

		elif x - IDENTIFIED_IMG_SIZE > 0:

			text_location_y = y + 20 + (index + 1) * 20
			text_location_x = x - IDENTIFIED_IMG_SIZE

			if text_location_y <= y + h:
				cv2.putText(
					img,
					emotion_label,
					(text_location_x, text_location_y),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.5,
					(255, 255, 255),
					1,
				)

				cv2.rectangle(
					img,
					(
						x - IDENTIFIED_IMG_SIZE + 70,
						y + 13 + (index + 1) * 20,
					),
					(
						x - IDENTIFIED_IMG_SIZE + 70 + bar_x,
						y + 13 + (index + 1) * 20 + 5,
					),
					(255, 255, 255),
					cv2.FILLED,
				)

	return img

def overlay_age_gender(
	img: np.ndarray, apparent_age: float, gender: str, x: int, y: int, w: int, h: int
) -> np.ndarray:
	"""
	Overlay the analyzed age and gender of face onto image itself
	Args:
		img (np.ndarray): image itself
		apparent_age (float): analyzed apparent age
		gender (str): analyzed gender
		x (int): x coordinate of the face on the given image
		y (int): y coordinate of the face on the given image
		w (int): w coordinate of the face on the given image
		h (int): h coordinate of the face on the given image
	Returns:
		img (np.ndarray): image with overlay age and gender analsis results
	"""
	logger.debug(f"{apparent_age} years old {gender}")
	analysis_report = f"{int(apparent_age)} {gender}"

	info_box_color = (46, 200, 255)

	# show its age and gender on the top of the image
	if y - IDENTIFIED_IMG_SIZE + int(IDENTIFIED_IMG_SIZE / 5) > 0:

		triangle_coordinates = np.array(
			[
				(x + int(w / 2), y),
				(
					x + int(w / 2) - int(w / 10),
					y - int(IDENTIFIED_IMG_SIZE / 3),
				),
				(
					x + int(w / 2) + int(w / 10),
					y - int(IDENTIFIED_IMG_SIZE / 3),
				),
			]
		)

		cv2.drawContours(
			img,
			[triangle_coordinates],
			0,
			info_box_color,
			-1,
		)

		cv2.rectangle(
			img,
			(
				x + int(w / 5),
				y - IDENTIFIED_IMG_SIZE + int(IDENTIFIED_IMG_SIZE / 5),
			),
			(x + w - int(w / 5), y - int(IDENTIFIED_IMG_SIZE / 3)),
			info_box_color,
			cv2.FILLED,
		)

		cv2.putText(
			img,
			analysis_report,
			(x + int(w / 3.5), y - int(IDENTIFIED_IMG_SIZE / 2.1)),
			cv2.FONT_HERSHEY_SIMPLEX,
			1,
			(0, 111, 255),
			2,
		)

	# show its age and gender on the top of the image
	elif y + h + IDENTIFIED_IMG_SIZE - int(IDENTIFIED_IMG_SIZE / 5) < img.shape[0]:

		triangle_coordinates = np.array(
			[
				(x + int(w / 2), y + h),
				(
					x + int(w / 2) - int(w / 10),
					y + h + int(IDENTIFIED_IMG_SIZE / 3),
				),
				(
					x + int(w / 2) + int(w / 10),
					y + h + int(IDENTIFIED_IMG_SIZE / 3),
				),
			]
		)

		cv2.drawContours(
			img,
			[triangle_coordinates],
			0,
			info_box_color,
			-1,
		)

		cv2.rectangle(
			img,
			(x + int(w / 5), y + h + int(IDENTIFIED_IMG_SIZE / 3)),
			(
				x + w - int(w / 5),
				y + h + IDENTIFIED_IMG_SIZE - int(IDENTIFIED_IMG_SIZE / 5),
			),
			info_box_color,
			cv2.FILLED,
		)

		cv2.putText(
			img,
			analysis_report,
			(x + int(w / 3.5), y + h + int(IDENTIFIED_IMG_SIZE / 1.5)),
			cv2.FONT_HERSHEY_SIMPLEX,
			1,
			(0, 111, 255),
			2,
		)

	return img

def analysis(model_name="VGG-Face",detector_backend="opencv",enable_face_analysis=True,source=0,time_threshold=5,frame_threshold=5,anti_spoofing: bool = False,):
	"""
	Run real time face recognition and facial attribute analysis

	Args:
		db_path (string): Path to the folder containing image files. All detected faces
			in the database will be considered in the decision-making process.

		model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
			OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

		detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
			'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
			(default is opencv).

		distance_metric (string): Metric for measuring similarity. Options: 'cosine',
			'euclidean', 'euclidean_l2' (default is cosine).

		enable_face_analysis (bool): Flag to enable face analysis (default is True).

		source (Any): The source for the video stream (default is 0, which represents the
			default camera).

		time_threshold (int): The time threshold (in seconds) for face recognition (default is 5).

		frame_threshold (int): The frame threshold for face recognition (default is 5).

		anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

	Returns:
		None
	"""
	# initialize models
	build_demography_models(enable_face_analysis=enable_face_analysis)
	build_facial_recognition_model(model_name=model_name)
	# call a dummy find function for db_path once to create embeddings before starting webcam
	#_ = search_identity(detected_face=np.zeros([224, 224, 3]),db_path=db_path,detector_backend=detector_backend,distance_metric=distance_metric,model_name=model_name,)

	freezed_img = None
	freeze = False
	num_frames_with_faces = 0
	tic = time.time()

	cap = cv2.VideoCapture(source)  # webcam
	while (countClipAddedToPlayList < numScenes):
		has_frame, img = cap.read()
		if not has_frame:
			break

		# we are adding some figures into img such as identified facial image, age, gender
		# that is why, we need raw image itself to make analysis
		raw_img = img.copy()

		faces_coordinates = []
		if freeze is False:
			faces_coordinates = grab_facial_areas(
				img=img, detector_backend=detector_backend, anti_spoofing=anti_spoofing
			)

			# we will pass img to analyze modules (identity, demography) and add some illustrations
			# that is why, we will not be able to extract detected face from img clearly
			detected_faces = extract_facial_areas(img=img, faces_coordinates=faces_coordinates)

			img = highlight_facial_areas(img=img, faces_coordinates=faces_coordinates)
			"""
			img = countdown_to_freeze(
				img=img,
				faces_coordinates=faces_coordinates,
				frame_threshold=frame_threshold,
				num_frames_with_faces=num_frames_with_faces,
			)
			"""

			num_frames_with_faces = num_frames_with_faces + 1 if len(faces_coordinates) else 0

			freeze = num_frames_with_faces > 0 and num_frames_with_faces % frame_threshold == 0
			if freeze:
				# add analyze results into img - derive from raw_img
				img = highlight_facial_areas(
					img=raw_img, faces_coordinates=faces_coordinates, anti_spoofing=anti_spoofing
				)

				# age, gender and emotion analysis
				img = perform_demography_analysis(
					enable_face_analysis=enable_face_analysis,
					img=raw_img,
					faces_coordinates=faces_coordinates,
					detected_faces=detected_faces,
				)
				# facial recogntion analysis
				#img = perform_facial_recognition(img=img,faces_coordinates=faces_coordinates,detected_faces=detected_faces,db_path=db_path,detector_backend=detector_backend,distance_metric=distance_metric,model_name=model_name,)

				# freeze the img after analysis
				freezed_img = img.copy()

				# start counter for freezing
				tic = time.time()
				logger.info("freezed")

		elif freeze is True and time.time() - tic > time_threshold:
			freeze = False
			freezed_img = None
			# reset counter for freezing
			tic = time.time()
			logger.info("freeze released")

		freezed_img = countdown_to_release(img=freezed_img, tic=tic, time_threshold=time_threshold)

		checkClipToAddToPlayList()
		
		cv2.imshow("img", img if freezed_img is None else freezed_img)

		if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
			break

	# kill open cv things
	cap.release()
	cv2.destroyAllWindows()

###################################### FIN ANALISIS ###################################################


###################################### TESTING ########################################################
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
  "GhostFaceNet",
]

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'fastmtcnn',
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'centerface',
]

#addClips()

#testClipsChosen()

#drawGraphFinalDecisionChosen()

addClips()

startClip()

analysis(models[2], detector_backend=backends[5],enable_face_analysis=True,source=0, time_threshold=0.1, frame_threshold=1)

finalDecision = getFinalDecisionChosen()

drawGraphFinalDecisionChosen()

print(finalDecision)

input("Press Enter to continue...")
###################################### TESTING ########################################################