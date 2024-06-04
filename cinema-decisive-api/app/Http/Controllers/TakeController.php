<?php

namespace App\Http\Controllers;

use App\Models\Take;
use App\Models\TakeDto;
use App\Models\TakesDto;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;
use App\Models\EmotionScore;

class TakeController extends Controller
{
    public function __construct()
    {

    }

    public function show(){
        return Take::all();
    }

    public function showTake($id){
        return Take::where('id','=',$id)->firstOrFail();
    }

    public function showTakes($idScene){
        return Take::where('id_scene','=',$idScene)->firstOrFail();
    }

    public function showTakesByMovie($idMovie){
        try{
            $takes = DB::table('take')
            ->join('scene', 'scene.id', '=', 'take.id_scene')
            ->join('emotion_score', 'emotion_score.id_take', '=', 'take.id')
            ->join('emotion', 'emotion_score.id_emotion', '=', 'emotion.id')
            ->where('scene.id_movie','=',$idMovie)
            ->select('take.*','emotion_score.id as id_emotion_score','emotion_score.score','emotion.name', 'emotion.id as id_emotion')
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
            $takesDto = new TakesDto();
            for($i=0;$i<count($takes);$i++){
                $take = $takes[$i];

                $id = $take->id;
                $id_father = $take->id_father;
                $video_path = $take->video_path;
                $image_path = $take->image_path;
                $id_scene = $take->id_scene;
                $duration = $take->duration;
                $name = $take->name;
                $score = $take->score;
                $id_emotion = $take->id_emotion;
                $id_emotion_score = $take->id_emotion_score;

                $takesFromTake = $takesDto->takesDto;

                $takeFound = $this->findObjectById($id, $takesFromTake);

                if($takeFound === false){
                    $takeDTO = new TakeDto($id,
                        $id_father,
                        $video_path,
                        $image_path,
                        $id_scene,
                        $duration);
                    $takeDTO->addEmotion(["id" => $id_emotion_score, "id_take" => $id, "id_emotion" => $id_emotion, "name" => $name, "score" => $score]);
                    $takesDto->addTake($takeDTO);
                }else{
                    $takeFound->addEmotion(["id" => $id_emotion_score, "id_take" => $id, "id_emotion" => $id_emotion, "name" => $name, "score" => $score]);
                }
            }
            return $takesDto->takesDto;
        } catch (\Exception $exception) {
            error_log($exception);
            return response()->json(['error'=> $exception->getMessage()],400);
        }
    }

    private function findObjectById($id, $array){
        foreach ( $array as $element ) {
            if ( $id == $element->id) {
                return $element;
            }
        }
        return false;
    }

    public function create(Request $request){
        try{
            $take = json_decode($request->take);

            $takeCreated = Take::create([
                'id_father'=> $take->id_father,
                'id_scene'=> $take->id_scene,
                'duration'=> $take->duration
            ]);

            $imageObj = $this->saveImage($request, $takeCreated->id);

            if(isset($imageObj["image"])){
                $takeCreated->image_path = "movies/".$request->id."_". $imageObj["path"];
                $takeCreated->save();
            }

            $videoObj = $this->saveVideo($request, $takeCreated->id);

            if(isset($videoObj["video"])){
                $takeCreated->video_path = "movies/".$request->id."_". $videoObj["path"];
                $takeCreated->save();
            }

            $this->saveEmotions($take->emotions);

            return $takeCreated;
        } catch (\Exception $exception) {
            error_log($exception);
            return response()->json(['error'=> $exception->getMessage()],400);
        }
    }

    public function update(Request $request){
        try{
            $take = json_decode($request->take);
            $movie = Take::where('id',$take->id)->firstOrFail();
            $movie->id_father = $take->id_father;
            $movie->duration = $take->duration;
            $movie->id_scene = $take->id_scene;

            $imageObj = $this->saveImage($request, $movie->id);

            if(isset($imageObj["image"])){
                $movie->image_path = "movies/".$take->id."_". $imageObj["path"];
            }

            $videoObj = $this->saveVideo($request, $movie->id);

            if(isset($videoObj["video"])){
                $movie->video_path = "movies/".$take->id."_". $videoObj["path"];
            }

            $this->saveEmotions($take->emotions);

            return $movie->save();
        } catch (\Exception $exception) {
            error_log($exception);
            return response()->json(['error'=> $exception->getMessage()],400);
        }
    }

    public function delete($id){
        try{
            return DB::table('take')->where('id', '=', $id)->delete();
        }catch(\Exception $exception){
            error_log($exception);
        }
    }

    private function saveEmotions($emotions){
        foreach($emotions as $emotion){
            if(property_exists($emotion,"id") && $emotion->id){
                $this->updateEmotionScore($emotion);
            }else{
                $this->createEmotionScore($emotion);
            }
        }
    }

    private function createEmotionScore($emotion){
        EmotionScore::create([
            'id_emotion'=> $emotion->id_emotion,
            'id_take'=> $emotion->id_take,
            'score'=> $emotion->score,
        ]);
    }

    private function updateEmotionScore($emotion){
        $movie = EmotionScore::where('id',$emotion->id)->first();
        $movie->id_emotion = $emotion->id_emotion;
        $movie->id_take = $emotion->id_take;
        $movie->score = $emotion->score;
        return $movie->save();
    }



    private function saveImage(Request $request, $idMovie){
        $image = $request->file('image');
        if(isset($image)){
            $name = $image->getClientOriginalName();
            $image->store('public/uploads');
            $image->move(public_path('movies'), $idMovie."_". $name);
            $path = $name;
            return ["image"=>$image, "path" => $path, "name" => $name];
        }
       return [];
    }

    private function saveVideo(Request $request, $idMovie){
        $video = $request->file('video');
        if(isset($video)){
            $name = $video->getClientOriginalName();
            $video->store('public/uploads');
            $video->move(public_path('movies'), $idMovie."_". $name);
            $path = $name;
            return ["movie"=>$video, "path" => $path, "name" => $name];
        }
       return [];
    }
}
