<?php

namespace App\Http\Controllers;

use App\Models\Emotion;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;

class EmotionController extends Controller
{
    public function __construct()
    {

    }

    public function show(){
        return Emotion::all();
    }

    public function create(Request $request){
        return Emotion::create([
            'name'=> $request->name,
            'description'=> $request->description,
        ]);
    }

    public function update(Request $request){
        $movie = Emotion::where('id',$request->id)->firstOrFail();
        $movie->name = $request->name;
        $movie->description = $request->description;
        return $movie->save();
    }

    public function delete($id){
        try{
            return DB::table('emotion')->where('id', '=', $id)->delete();
        }catch(\Exception $exception){
            error_log($exception);
        }
    }
}
