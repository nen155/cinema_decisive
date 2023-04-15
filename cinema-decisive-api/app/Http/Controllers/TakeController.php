<?php

namespace App\Http\Controllers;

use App\Models\Take;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;

class TakeController extends Controller
{
    public function __construct()
    {

    }

    public function show(){
        return Take::all();
    }

    public function showTakes($idScene){
        return Take::where('id_scene',$idScene);
    }

    public function showTakesByMovie($idMovie){
        try{
            $takes = DB::table('take')
            ->join('scene', 'scene.id', '=', 'take.id_scene')
            ->join('emotion_score', 'emotion_score.id_take', '=', 'take.id')
            ->join('emotion', 'emotion_score.id_emotion', '=', 'emotion.id')
            ->where('scene.id_movie','=',$idMovie)
            ->select('take.*','emotion_score.score','emotion.name')
            ->get();
            $result = $this->toDto($takes->toArray());
            return $result;
        } catch (\Exception $exception) {
            error_log($exception);
            return response()->json(['error'=> $exception->getMessage()],400);
        }
    }

    private function toDto($takes){
        try{
            $takesDto = array();
            for($i=0;$i<count($takes);$i++){
                $take = $takes[$i];

                $id = $take->id;
                $id_father = $take->id_father;
                $videp_path = $take->videp_path;
                $id_scene = $take->id_scene;
                $duration = $take->duration;
                $name = $take->name;
                $score = $take->score;

                error_log($id);
                $takeFound = $this->findObjectById($id, $takesDto);
                if($takeFound == false){
                    $takeDTO = [
                        "id" => $id,
                        "id_father" => $id_father,
                        "videp_path" => $videp_path,
                        "id_scene" => $id_scene,
                        "duration" => $duration,
                        "emotions" =>array($name)
                            ];
                    $takesDto[] = $takeDTO;
                }else{
                    $emotions = $takeFound["emotions"];
                    $emotions[] = $name;
                }
            }
            return $takesDto;
        } catch (\Exception $exception) {
            error_log($exception);
            return response()->json(['error'=> $exception->getMessage()],400);
        }
    }

    private function findObjectById($id, $array){
        foreach ( $array as $element ) {
            if ( $id == $element["id"] ) {
                return $element;
            }
        }
        return false;
    }

    public function create(Request $request){
        return Take::create([
            'id_father'=> $request->id_father,
            'videp_path'=> $request->videp_path,
            'id_scene'=> $request->id_scene,
            'duration'=> $request->duration,
        ]);
    }

    public function update(Request $request){
        $movie = Take::where('id',$request->id)->firstOrFail();
        $movie->id_father = $request->id_father;
        $movie->videp_path = $request->videp_path;
        $movie->duration = $request->duration;
        $movie->id_scene = $request->id_scene;
        return $movie->save();
    }

    public function delete($id){
        try{
            return DB::table('take')->where('id', '=', $id)->delete();
        }catch(\Exception $exception){
            error_log($exception);
        }
    }
}
