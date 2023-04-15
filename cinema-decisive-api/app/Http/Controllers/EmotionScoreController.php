<?php

namespace App\Http\Controllers;

use App\Models\EmotionScore;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;

class EmotionScoreController extends Controller
{
    public function __construct()
    {

    }

    public function show(){
        return EmotionScore::all();
    }

    public function showEmotions($idTake){
        return EmotionScore::where('id_take',$idTake);
    }


    public function create(Request $request){
        return EmotionScore::create([
            'id_emotion'=> $request->id_emotion,
            'id_take'=> $request->id_take,
            'score'=> $request->score,
        ]);
    }

    public function update(Request $request){
        $movie = EmotionScore::where('id',$request->id)->firstOrFail();
        $movie->id_emotion = $request->id_emotion;
        $movie->id_take = $request->id_take;
        $movie->score = $request->score;
        return $movie->save();
    }

    public function delete($id){
        try{
            return DB::table('emotion_score')->where('id', '=', $id)->delete();
        }catch(\Exception $exception){
            error_log($exception);
        }
    }
}
