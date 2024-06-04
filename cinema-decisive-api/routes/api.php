<?php


use App\Http\Controllers\EmotionController;
use App\Http\Controllers\EmotionScoreController;
use App\Http\Controllers\TakeController;
use App\Http\Controllers\MovieController;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;

/*
|--------------------------------------------------------------------------
| API Routes
|--------------------------------------------------------------------------
|
| Here is where you can register API routes for your application. These
| routes are loaded by the RouteServiceProvider within a group which
| is assigned the "api" middleware group. Enjoy building your API!
|
*/

Route::middleware('auth:sanctum')->get('/user', function (Request $request) {
    return $request->user();
});

Route::get('movies', [MovieController::class,'show']);
Route::get('movie/{id}', [MovieController::class,'showMovie']);
Route::post('movie', [MovieController::class,'create']);
Route::post('movie-update', [MovieController::class,'update']);
Route::get('movie-delete/{idMovie}', [MovieController::class,'delete']);

Route::get('take', [TakeController::class,'show']);
Route::get('take/{id}', [TakeController::class,'showTake']);
Route::get('take-scene/{idScene}', [TakeController::class,'showTakes']);
Route::get('take-movie/{idMovie}', [TakeController::class,'showTakesByMovie']);
Route::post('take', [TakeController::class,'create']);
Route::post('take-update', [TakeController::class,'update']);
Route::get('take-delete/{idTake}', [TakeController::class,'delete']);

Route::get('emotion', [EmotionController::class,'show']);
Route::post('emotion', [EmotionController::class,'create']);
Route::post('emotion-update', [EmotionController::class,'update']);
Route::get('emotion-delete/{idEmotion}', [EmotionController::class,'delete']);

Route::get('emotion-score', [EmotionScoreController::class,'show']);
Route::get('emotion-score/{idTake}', [EmotionScoreController::class,'showEmotions']);
Route::post('emotion-score', [EmotionScoreController::class,'create']);
Route::post('emotion-score-update', [EmotionScoreController::class,'update']);
Route::get('emotion-score-delete/{idEmotionScore}', [EmotionScoreController::class,'delete']);

Route::get('scene', [SceneController::class,'show']);
Route::get('scene/{idMovie}', [SceneController::class,'showScenes']);
Route::post('scene', [SceneController::class,'create']);
Route::post('scene-update', [SceneController::class,'update']);
Route::get('scene-delete/{idScene}', [SceneController::class,'delete']);
