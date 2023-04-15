-- phpMyAdmin SQL Dump
-- version 5.1.1
-- https://www.phpmyadmin.net/
--
-- Servidor: 127.0.0.1
-- Tiempo de generación: 21-03-2023 a las 17:53:23
-- Versión del servidor: 10.4.21-MariaDB
-- Versión de PHP: 8.0.10

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Base de datos: `cinema_decisive`
--

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `emotion`
--

CREATE TABLE `emotion` (
  `id` int(11) NOT NULL,
  `name` varchar(200) NOT NULL,
  `description` varchar(1000) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Volcado de datos para la tabla `emotion`
--

INSERT INTO `emotion` (`id`, `name`, `description`) VALUES
(1, 'Angry', ''),
(2, 'Happy', ''),
(3, 'Disgust', ''),
(4, 'Fear', ''),
(5, 'Surprise', ''),
(6, 'Sad', ''),
(7, 'Neutral', '');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `emotion_score`
--

CREATE TABLE `emotion_score` (
  `id` int(11) NOT NULL,
  `id_emotion` int(11) NOT NULL,
  `id_take` int(11) NOT NULL,
  `score` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Volcado de datos para la tabla `emotion_score`
--

INSERT INTO `emotion_score` (`id`, `id_emotion`, `id_take`, `score`) VALUES
(1, 1, 1, 5),
(2, 2, 1, 6);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `movie`
--

CREATE TABLE `movie` (
  `id` int(11) NOT NULL,
  `name` varchar(200) NOT NULL,
  `duration` int(11) NOT NULL,
  `image_path` varchar(500) NOT NULL,
  `image_name` varchar(500) DEFAULT NULL,
  `number_scenes` int(11) NOT NULL,
  `created_at` datetime DEFAULT NULL,
  `updated_at` datetime DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Volcado de datos para la tabla `movie`
--

INSERT INTO `movie` (`id`, `name`, `duration`, `image_path`, `image_name`, `number_scenes`, `created_at`, `updated_at`) VALUES
(1, 'Jurassic Park', 120, 'movies/prueba.jpg', 'prueba.jpg', 12, NULL, '2023-03-15 18:47:10'),
(2, 'Seat of the Synod', 120, 'movies/2_prueba2.jpg', 'prueba2.jpg', 12, '2023-01-15 21:48:50', '2023-03-15 19:23:12');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `scene`
--

CREATE TABLE `scene` (
  `id` int(11) NOT NULL,
  `id_emotion_base` int(11) NOT NULL,
  `name` varchar(200) NOT NULL,
  `id_movie` int(11) NOT NULL,
  `duration` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Volcado de datos para la tabla `scene`
--

INSERT INTO `scene` (`id`, `id_emotion_base`, `name`, `id_movie`, `duration`) VALUES
(1, 1, '', 1, 20),
(2, 1, '', 1, 50);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `take`
--

CREATE TABLE `take` (
  `id` int(11) NOT NULL,
  `id_father` int(11) DEFAULT NULL,
  `video_path` varchar(8000) NOT NULL,
  `id_scene` int(11) NOT NULL,
  `duration` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Volcado de datos para la tabla `take`
--

INSERT INTO `take` (`id`, `id_father`, `video_path`, `id_scene`, `duration`) VALUES
(1, NULL, 'movie/1.mpg', 1, 20);

--
-- Índices para tablas volcadas
--

--
-- Indices de la tabla `emotion`
--
ALTER TABLE `emotion`
  ADD PRIMARY KEY (`id`);

--
-- Indices de la tabla `emotion_score`
--
ALTER TABLE `emotion_score`
  ADD PRIMARY KEY (`id`);

--
-- Indices de la tabla `movie`
--
ALTER TABLE `movie`
  ADD PRIMARY KEY (`id`);

--
-- Indices de la tabla `scene`
--
ALTER TABLE `scene`
  ADD PRIMARY KEY (`id`);

--
-- Indices de la tabla `take`
--
ALTER TABLE `take`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT de las tablas volcadas
--

--
-- AUTO_INCREMENT de la tabla `emotion`
--
ALTER TABLE `emotion`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=8;

--
-- AUTO_INCREMENT de la tabla `emotion_score`
--
ALTER TABLE `emotion_score`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT de la tabla `movie`
--
ALTER TABLE `movie`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=5;

--
-- AUTO_INCREMENT de la tabla `scene`
--
ALTER TABLE `scene`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT de la tabla `take`
--
ALTER TABLE `take`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
