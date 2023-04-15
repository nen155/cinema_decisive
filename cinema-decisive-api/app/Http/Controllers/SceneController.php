<?php

namespace App\Http\Controllers;

use App\Models\Scene;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;

class SceneController extends Controller
{
    public function __construct()
    {

    }

    public function show(){
        return Scene::all();
    }

    public function showScenes($idMovie){
        return Scene::where('id_movie',$idMovie);
    }

    public function create(Request $request){
        return Scene::create([
            'id_emotion_base'=> $request->id_emotion_base,
            'id_movie'=> $request->id_movie,
            'duration'=> $request->duration,
        ]);
    }

    public function update(Request $request){
        $movie = Scene::where('id',$request->id)->firstOrFail();
        $movie->id_emotion_base = $request->id_emotion_base;
        $movie->id_movie = $request->id_movie;
        $movie->duration = $request->duration;
        return $movie->save();
    }

    public function delete($id){
        try{
            return DB::table('scene')->where('id', '=', $id)->delete();
        }catch(\Exception $exception){
            error_log($exception);
        }
    }
}
