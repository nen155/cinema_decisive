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

Route::get('movie', [MovieController::class,'show']);
Route::post('movie', [MovieController::class,'create']);
Route::post('movie-update', [MovieController::class,'update']);
Route::get('movie-delete/{idMovie}', [MovieController::class,'delete']);

Route::get('take', [TakeController::class,'show']);
Route::get('take/{idScene}', [TakeController::class,'showTakes']);
Route::get('take-movie/{idMovie}', [TakeController::class,'showTakesByMovie']);
Route::post('take', [TakeController::class,'create']);
Route::put('take', [TakeController::class,'update']);
Route::delete('take', [TakeController::class,'delete']);

Route::get('emotion', [EmotionController::class,'show']);
Route::post('emotion', [EmotionController::class,'create']);
Route::put('emotion', [EmotionController::class,'update']);
Route::delete('emotion', [EmotionController::class,'delete']);

Route::get('emotion-score', [EmotionScoreController::class,'show']);
Route::get('emotion-score/{idTake}', [EmotionScoreController::class,'showEmotions']);
Route::post('emotion-score', [EmotionScoreController::class,'create']);
Route::put('emotion-score', [EmotionScoreController::class,'update']);
Route::delete('emotion-score', [EmotionScoreController::class,'delete']);

Route::get('scene', [SceneController::class,'show']);
Route::get('scene/{idMovie}', [SceneController::class,'showScenes']);
Route::post('scene', [SceneController::class,'create']);
Route::put('scene', [SceneController::class,'update']);
Route::delete('scene', [SceneController::class,'delete']);
